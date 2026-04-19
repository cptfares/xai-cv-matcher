"""
Layer 4 — Skill Matching Model

Uses sentence-transformers to compute semantic similarity between CV skills
and job requirements.  Falls back to a fast TF-IDF cosine baseline if the
model is unavailable (offline / CI environments).

Outputs a MatchResult with:
  - per-skill similarity scores
  - weighted overall match score (0–100)
  - experience & seniority sub-scores
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from cv_matcher.models import CVFeatures, JobFeatures, MatchResult, SkillMatch

# ── Threshold ─────────────────────────────────────────────────────────────────
MATCH_THRESHOLD = 0.60      # similarity ≥ this → skill is considered matched
SYNONYM_THRESHOLD = 0.72    # semantic match but not exact-string → flag as synonym

# ── Embedding model (lazy, loaded once) ──────────────────────────────────────
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            warnings.warn(
                "sentence-transformers not available; falling back to TF-IDF.",
                RuntimeWarning,
            )
            _encoder = _TFIDFEncoder()
    return _encoder


# ── TF-IDF fallback ───────────────────────────────────────────────────────────
class _TFIDFEncoder:
    """Minimal TF-IDF character-ngram encoder as a zero-dep fallback."""

    def encode(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        try:
            return vec.fit_transform(texts).toarray().astype(np.float32)
        except Exception:
            # Return identity vectors if corpus is too small
            n = len(texts)
            return np.eye(n, dtype=np.float32)


# ── Seniority ladder ──────────────────────────────────────────────────────────
_SENIORITY_ORDER = ["junior", "mid", "senior", "lead"]


def _seniority_match_score(cv_level: str, job_level: str) -> float:
    """
    Returns a 0–1 score.  Exact match → 1.0.
    One level away → 0.7.  Two+ levels away → 0.4.
    """
    try:
        cv_idx = _SENIORITY_ORDER.index(cv_level)
        job_idx = _SENIORITY_ORDER.index(job_level)
    except ValueError:
        return 0.7  # unknown level → neutral
    diff = abs(cv_idx - job_idx)
    return [1.0, 0.7, 0.4, 0.2][min(diff, 3)]


class SkillMatcher:
    """
    Match a CVFeatures object against a JobFeatures object.

    The matching model is intentionally replaceable: swap _get_encoder()
    for any model that exposes an encode(list[str]) → ndarray interface.
    """

    def match(self, cv: CVFeatures, job: JobFeatures) -> MatchResult:
        job_skills = job.required_skills
        cv_skills = cv.normalized_skills

        if not job_skills:
            return MatchResult(overall_score=0.0)

        # ── Encode all skills ──────────────────────────────────────────────
        encoder = _get_encoder()
        all_texts = job_skills + cv_skills
        embeddings = encoder.encode(all_texts)
        job_embs = embeddings[: len(job_skills)]
        cv_embs = embeddings[len(job_skills):]

        # ── Similarity matrix: [job_skills × cv_skills] ───────────────────
        if cv_embs.shape[0] == 0:
            sim_matrix = np.zeros((len(job_skills), 1))
        else:
            sim_matrix = cosine_similarity(job_embs, cv_embs)

        # ── Per-skill best match ───────────────────────────────────────────
        skill_matches: list[SkillMatch] = []
        for i, js in enumerate(job_skills):
            weight = job.skill_weights.get(js, 1.0 / len(job_skills))

            if cv_skills:
                best_idx = int(np.argmax(sim_matrix[i]))
                best_score = float(sim_matrix[i][best_idx])
                best_cv_skill: Optional[str] = cv_skills[best_idx]
            else:
                best_score = 0.0
                best_cv_skill = None

            is_matched = best_score >= MATCH_THRESHOLD
            is_synonym = (
                is_matched
                and best_score < SYNONYM_THRESHOLD
                and best_cv_skill != js
            )

            skill_matches.append(
                SkillMatch(
                    job_skill=js,
                    cv_skill=best_cv_skill if is_matched else None,
                    similarity_score=round(best_score, 4),
                    weight=weight,
                    weighted_contribution=round(best_score * weight, 4),
                    is_matched=is_matched,
                    is_semantic_synonym=is_synonym,
                )
            )

        # ── Aggregate score ───────────────────────────────────────────────
        # Weighted sum of (similarity × weight), normalized to 0-100
        total_weight = sum(job.skill_weights.values()) or 1.0
        raw_skill_score = sum(m.weighted_contribution for m in skill_matches)
        skill_score = min(100.0, round(raw_skill_score / total_weight * 100, 1))

        # Experience score
        exp_score = self._experience_score(cv, job)

        # Seniority match
        seniority_ok = (
            _seniority_match_score(cv.seniority_level, job.experience_level) >= 0.7
        )

        # Blended overall: 70% skill, 20% experience, 10% seniority
        seniority_contribution = (
            _seniority_match_score(cv.seniority_level, job.experience_level) * 100
        )
        overall = round(
            0.70 * skill_score + 0.20 * exp_score + 0.10 * seniority_contribution, 1
        )

        return MatchResult(
            overall_score=overall,
            skill_matches=skill_matches,
            matched_count=sum(1 for m in skill_matches if m.is_matched),
            missing_count=sum(1 for m in skill_matches if not m.is_matched),
            total_required=len(skill_matches),
            experience_score=exp_score,
            seniority_match=seniority_ok,
        )

    # ── Sub-scores ────────────────────────────────────────────────────────────

    def _experience_score(self, cv: CVFeatures, job: JobFeatures) -> float:
        """
        Score 0–100 based on candidate years vs required years.
        Full marks if candidate meets or exceeds the requirement.
        Partial credit below.
        """
        required = job.required_experience_years
        if required is None or required == 0:
            return 80.0  # no stated requirement → neutral-high score

        candidate = cv.total_experience_years
        if candidate >= required:
            # Slight bonus for exceeding, capped at 100
            bonus = min((candidate - required) / required * 10, 10)
            return min(100.0, round(90 + bonus, 1))
        else:
            # Penalize proportionally for shortfall
            ratio = candidate / required
            return round(ratio * 90, 1)
