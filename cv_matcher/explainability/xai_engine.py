"""
Layer 5 — XAI Explainability Engine

Post-hoc explainability layer that works on top of any matching model output.

Produces:
  1. SHAP-inspired feature importances  — which skills helped / hurt the score
  2. Gap analysis                        — missing / weak skills with importance rank
  3. Bias flags                          — vocabulary-mismatch near-misses
  4. Natural language narratives          — for recruiters and candidates
"""
from __future__ import annotations
import logging

from cv_matcher.llm.client import get_client
from cv_matcher.llm.prompts import (
    SYSTEM_PROMPT,
    RECRUITER_NARRATIVE_USER,
    CANDIDATE_NARRATIVE_USER,
    UPSKILLING_ROADMAP_USER,
    SKILL_GAP_EXPLANATION_USER,
    BIAS_EXPLANATION_USER,
)
from cv_matcher.models import (
    BiasFlag,
    CVFeatures,
    GapItem,
    JobFeatures,
    MatchResult,
    SkillContribution,
    XAIReport,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
POSITIVE_THRESHOLD = 0.60    # similarity above this → positive contribution
BIAS_SIMILARITY_LOW = 0.45   # too low for a match but suspicious near-miss
BIAS_SIMILARITY_HIGH = 0.62  # above match threshold — no bias concern

_UPSKILL_RESOURCES: dict[str, str] = {
    "docker":         "Docker official docs (docs.docker.com) + 'Docker for Beginners' on YouTube",
    "kubernetes":     "Kubernetes.io interactive tutorials + CKAD/CKA certification path",
    "ci/cd":          "GitHub Actions quickstart or GitLab CI/CD documentation",
    "terraform":      "HashiCorp Learn platform (learn.hashicorp.com/terraform)",
    "aws":            "AWS Skill Builder free tier + AWS Solutions Architect Associate cert",
    "azure":          "Microsoft Learn (learn.microsoft.com) + AZ-900 foundational cert",
    "gcp":            "Google Cloud Skills Boost + Associate Cloud Engineer cert",
    "pytorch":        "fast.ai practical deep-learning course + PyTorch official tutorials",
    "tensorflow":     "TensorFlow Developer Certificate program",
    "react":          "React docs (react.dev) + Scrimba React course",
    "typescript":     "TypeScript Handbook (typescriptlang.org) + Execute Program",
    "graphql":        "How to GraphQL (howtographql.com)",
    "sql":            "Mode SQL Tutorial or SQLZoo interactive exercises",
    "postgresql":     "PostgreSQL official tutorial + pgExercises.com",
    "python":         "Python.org official tutorial + Real Python",
    "rust":           "The Rust Book (doc.rust-lang.org/book)",
    "go":             "A Tour of Go (go.dev/tour)",
    "kafka":          "Confluent Apache Kafka 101 free course",
    "spark":          "Databricks free training + Spark documentation",
    "mlflow":         "MLflow documentation + Databricks MLflow tutorials",
    "ansible":        "Red Hat Ansible Automation Platform learning path",
    "linux":          "Linux Foundation free courses + The Linux Command Line book",
}


class XAIEngine:
    """
    Post-hoc XAI layer — fully decoupled from the matching model.
    Accepts a MatchResult + features and returns an XAIReport.
    """

    def explain(
        self,
        match: MatchResult,
        cv_features: CVFeatures,
        job_features: JobFeatures,
    ) -> XAIReport:
        importances = self._compute_feature_importances(match)
        gaps = self._compute_gaps(match, job_features)
        bias_flags = self._detect_bias(match)
        narrative_r, narrative_c = self._generate_narratives(
            match, gaps, bias_flags, cv_features, job_features
        )

        return XAIReport(
            feature_importances=importances,
            gaps=gaps,
            bias_flags=bias_flags,
            narrative_recruiter=narrative_r,
            narrative_candidate=narrative_c,
        )

    # ── Feature importances (SHAP-inspired) ──────────────────────────────────

    def _compute_feature_importances(
        self, match: MatchResult
    ) -> list[SkillContribution]:
        """
        Approximate Shapley value per skill:
            shap_i = (similarity_i - baseline) × weight_i
        where baseline = overall_score / 100 (the "average" prediction without
        any single feature).

        Normalized so |shap| values sum to 1.
        """
        if not match.skill_matches:
            return []

        baseline = match.overall_score / 100.0
        raw: list[tuple[str, float]] = []

        for sm in match.skill_matches:
            delta = sm.similarity_score - baseline
            shap = round(delta * sm.weight, 4)
            raw.append((sm.job_skill, shap))

        # Normalize by absolute sum
        abs_sum = sum(abs(v) for _, v in raw) or 1.0
        contributions: list[SkillContribution] = []
        for skill, shap in sorted(raw, key=lambda x: -abs(x[1])):
            norm_shap = round(shap / abs_sum, 4)
            direction = (
                "positive" if norm_shap > 0.01
                else "negative" if norm_shap < -0.01
                else "neutral"
            )
            explanation = self._shap_explanation(skill, norm_shap, match)
            contributions.append(
                SkillContribution(
                    skill=skill,
                    shap_value=norm_shap,
                    direction=direction,
                    explanation=explanation,
                )
            )

        return contributions

    def _shap_explanation(
        self, skill: str, shap: float, match: MatchResult
    ) -> str:
        sm_map = {m.job_skill: m for m in match.skill_matches}
        sm = sm_map.get(skill)
        if sm is None:
            return f"'{skill}' had no impact on the score."

        pct = round(sm.similarity_score * 100)
        if sm.is_matched and not sm.is_semantic_synonym:
            return (
                f"'{skill}' is a strong match ({pct}% similarity), "
                f"contributing positively (SHAP: {shap:+.3f})."
            )
        elif sm.is_semantic_synonym:
            return (
                f"'{skill}' was matched semantically via '{sm.cv_skill}' "
                f"({pct}% similarity). Slight vocabulary mismatch detected."
            )
        else:
            return (
                f"'{skill}' is missing from the CV "
                f"(best similarity: {pct}%), pulling the score down "
                f"(SHAP: {shap:+.3f})."
            )

    # ── Gap analysis ──────────────────────────────────────────────────────────

    def _compute_gaps(
        self, match: MatchResult, job: JobFeatures
    ) -> list[GapItem]:
        gaps: list[GapItem] = []
        for sm in match.skill_matches:
            if not sm.is_matched:
                importance = round(sm.weight, 3)
                suggestion = _UPSKILL_RESOURCES.get(sm.job_skill, "")
                if not suggestion:
                    suggestion = (
                        f"Study '{sm.job_skill}' via official documentation, "
                        f"online courses (Udemy, Coursera), or hands-on projects."
                    )
                gaps.append(
                    GapItem(
                        missing_skill=sm.job_skill,
                        importance=importance,
                        upskilling_suggestion=suggestion,
                    )
                )
        # Sort by importance descending
        return sorted(gaps, key=lambda g: -g.importance)

    # ── Bias detection ────────────────────────────────────────────────────────

    def _detect_bias(self, match: MatchResult) -> list[BiasFlag]:
        """
        Flag cases where a CV skill and a job skill are near-synonyms
        but fell below the match threshold due to vocabulary mismatch.
        E.g., 'front-end development' vs 'react' — semantically close
        but lexically different.
        """
        flags: list[BiasFlag] = []
        for sm in match.skill_matches:
            if (
                not sm.is_matched
                and sm.cv_skill is not None
                and BIAS_SIMILARITY_LOW <= sm.similarity_score < BIAS_SIMILARITY_HIGH
            ):
                flags.append(
                    BiasFlag(
                        cv_skill=sm.cv_skill,
                        job_skill=sm.job_skill,
                        similarity=sm.similarity_score,
                        note=(
                            f"'{sm.cv_skill}' and '{sm.job_skill}' share "
                            f"{round(sm.similarity_score * 100)}% semantic similarity "
                            f"but fall below the match threshold ({round(BIAS_SIMILARITY_HIGH * 100)}%). "
                            f"This may penalize synonymous vocabulary. "
                            f"Consider reviewing manually."
                        ),
                    )
                )
        return flags

    # ── Natural language narratives ───────────────────────────────────────────

    def _generate_narratives(
        self,
        match: MatchResult,
        gaps: list[GapItem],
        bias_flags: list[BiasFlag],
        cv: CVFeatures,
        job: JobFeatures,
    ) -> tuple[str, str]:
        matched = [m for m in match.skill_matches if m.is_matched]
        missing_names = [g.missing_skill for g in gaps[:5]]
        matched_names = [m.job_skill for m in matched[:8]]
        top_gap = gaps[0] if gaps else None

        llm = get_client()
        if llm.available:
            recruiter_narrative = self._llm_recruiter_narrative(
                match, matched_names, missing_names, cv, job, bias_flags, llm
            )
            candidate_narrative = self._llm_candidate_narrative(
                match, matched_names, missing_names, cv, job, top_gap, bias_flags, llm
            )
            # Fall back to template only if LLM returned empty
            if not recruiter_narrative:
                recruiter_narrative = self._template_recruiter(match, matched_names, missing_names, cv, job, bias_flags)
            if not candidate_narrative:
                candidate_narrative = self._template_candidate(match, matched_names, missing_names, cv, job, top_gap, bias_flags)
        else:
            recruiter_narrative = self._template_recruiter(match, matched_names, missing_names, cv, job, bias_flags)
            candidate_narrative = self._template_candidate(match, matched_names, missing_names, cv, job, top_gap, bias_flags)

        return recruiter_narrative, candidate_narrative

    def _llm_recruiter_narrative(self, match, matched_names, missing_names, cv, job, bias_flags, llm) -> str:
        required_years = f"{job.required_experience_years:.0f}+" if job.required_experience_years else "not specified"
        domain_match = "aligned" if cv.domain == job.domain else f"candidate domain is {cv.domain or 'general'}, role is {job.domain or 'general'}"
        user = RECRUITER_NARRATIVE_USER.format(
            score=f"{match.overall_score:.0f}",
            matched_count=match.matched_count,
            total_required=match.total_required,
            matched_skills=", ".join(matched_names) or "none",
            missing_skills=", ".join(missing_names) or "none",
            candidate_years=f"{cv.total_experience_years:.1f}",
            required_years=required_years,
            cv_seniority=cv.seniority_level or "unknown",
            job_seniority=job.experience_level or "unknown",
            domain_match=domain_match,
            bias_count=len(bias_flags),
        )
        return llm.complete(SYSTEM_PROMPT, user)

    def _llm_candidate_narrative(self, match, matched_names, missing_names, cv, job, top_gap, bias_flags, llm) -> str:
        required_years = f"{job.required_experience_years:.0f}+" if job.required_experience_years else "not specified"
        top_tip = top_gap.upskilling_suggestion if top_gap else "keep building on your existing strengths"
        user = CANDIDATE_NARRATIVE_USER.format(
            score=f"{match.overall_score:.0f}",
            matched_count=match.matched_count,
            total_required=match.total_required,
            matched_skills=", ".join(matched_names) or "none identified",
            missing_skills=", ".join(missing_names) or "none — great fit!",
            candidate_years=f"{cv.total_experience_years:.1f}",
            required_years=required_years,
            cv_seniority=cv.seniority_level or "unknown",
            job_seniority=job.experience_level or "unknown",
            top_upskill_tip=top_tip,
        )
        return llm.complete(SYSTEM_PROMPT, user)

    def _template_recruiter(self, match, matched_names, missing_names, cv, job, bias_flags) -> str:
        """Fallback string-template recruiter narrative (no LLM)."""
        score = match.overall_score
        parts = [
            f"Overall match score: {score:.0f}/100 ({match.matched_count}/{match.total_required} required skills matched)."
        ]
        if matched_names:
            parts.append(f"The candidate demonstrates proficiency in: {', '.join(matched_names)}.")
        exp = self._experience_comment(cv, job, recruiter=True)
        if exp:
            parts.append(exp)
        if missing_names:
            parts.append(f"Key gaps to consider: {', '.join(missing_names)}.")
        if score >= 80:
            parts.append("Recommendation: Strong candidate — consider advancing to the interview stage.")
        elif score >= 60:
            parts.append("Recommendation: Moderate match. Conduct a technical screen to validate capabilities.")
        else:
            parts.append("Recommendation: Significant skill gaps present. Consider only if the role allows a learning runway.")
        if bias_flags:
            parts.append(f"Note: {len(bias_flags)} vocabulary-mismatch flag(s) detected — review manually.")
        return " ".join(parts)

    def _template_candidate(self, match, matched_names, missing_names, cv, job, top_gap, bias_flags) -> str:
        """Fallback string-template candidate narrative (no LLM)."""
        score = match.overall_score
        parts = [f"Your CV scored {score:.0f}/100 for this role. You match {match.matched_count} of {match.total_required} required skills."]
        if matched_names:
            parts.append(f"Your strengths align well with: {', '.join(matched_names)}.")
        exp = self._experience_comment(cv, job, recruiter=False)
        if exp:
            parts.append(exp)
        if missing_names:
            parts.append(f"To improve your match, consider developing: {', '.join(missing_names)}.")
        if top_gap and top_gap.upskilling_suggestion:
            parts.append(f"For '{top_gap.missing_skill}': {top_gap.upskilling_suggestion}")
        if bias_flags:
            bias_skills = ", ".join(f"'{b.cv_skill}'" for b in bias_flags[:3])
            parts.append(f"Your CV mentions {bias_skills} which may be semantically close to job requirements — consider using the exact terms from the job description.")
        return " ".join(parts)

    def _experience_comment(
        self, cv: CVFeatures, job: JobFeatures, recruiter: bool
    ) -> str:
        required = job.required_experience_years
        candidate = cv.total_experience_years
        if required is None:
            return ""
        if recruiter:
            if candidate >= required:
                return (
                    f"Experience: Candidate has {candidate:.1f} years "
                    f"(requirement: {required:.0f}+ years) — meets or exceeds threshold."
                )
            else:
                shortfall = required - candidate
                return (
                    f"Experience: Candidate has {candidate:.1f} years; "
                    f"role requires {required:.0f}+ years "
                    f"({shortfall:.1f} year shortfall)."
                )
        else:
            if candidate >= required:
                return (
                    f"Your {candidate:.1f} years of experience meets "
                    f"the {required:.0f}+ year requirement."
                )
            else:
                return (
                    f"The role requires {required:.0f}+ years of experience; "
                    f"you currently have {candidate:.1f}. "
                    f"Highlighting relevant project depth can help offset this gap."
                )
