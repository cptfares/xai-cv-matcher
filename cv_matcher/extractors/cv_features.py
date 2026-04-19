"""
Layer 3 — CV Feature Extractor

Normalizes parsed CV data into structured features used by the matching layer.
"""
from __future__ import annotations

import re
from cv_matcher.models import CVFeatures, ParsedCV

_SENIORITY_MAP: dict[str, list[str]] = {
    "junior": ["junior", "jr", "entry", "intern", "trainee", "graduate"],
    "mid":    ["mid", "associate", "intermediate"],
    "senior": ["senior", "sr", "principal"],
    "lead":   ["lead", "staff", "head", "director", "manager", "architect", "vp"],
}

_DOMAIN_VOCAB: dict[str, set[str]] = {
    "frontend":  {"react", "vue", "angular", "html", "css", "typescript", "next.js"},
    "backend":   {"django", "flask", "fastapi", "spring", "node.js", "express", "ruby"},
    "data":      {"sql", "pandas", "spark", "hadoop", "airflow", "dbt", "r", "matlab"},
    "ml_ai":     {"tensorflow", "pytorch", "scikit-learn", "mlflow", "llm", "keras"},
    "devops":    {"docker", "kubernetes", "terraform", "ansible", "ci/cd", "jenkins", "helm"},
    "fullstack": {"react", "node.js", "postgresql", "mongodb", "docker", "typescript"},
    "mobile":    {"swift", "kotlin", "flutter", "react native"},
}


class CVFeatureExtractor:
    """Convert a ParsedCV into a CVFeatures object."""

    def extract(self, cv: ParsedCV) -> CVFeatures:
        normalized_skills = self._normalize_skills(cv.skills)
        skill_exp = self._estimate_skill_experience(cv)
        seniority = self._infer_seniority(cv)
        domain = self._infer_domain(normalized_skills)

        return CVFeatures(
            normalized_skills=normalized_skills,
            skill_experience_months=skill_exp,
            seniority_level=seniority,
            domain=domain,
            total_experience_years=cv.total_experience_years,
        )

    # ── Normalization ─────────────────────────────────────────────────────────

    def _normalize_skills(self, skills: list[str]) -> list[str]:
        """Lower-case, deduplicate, apply canonical aliases."""
        aliases: dict[str, str] = {
            "react.js": "react",
            "reactjs": "react",
            "node": "node.js",
            "nodejs": "node.js",
            "postgres": "postgresql",
            "k8s": "kubernetes",
            "vue.js": "vue",
            "vuejs": "vue",
            "next": "next.js",
            "nuxt.js": "nuxt",
            "typescript": "typescript",
            "js": "javascript",
            "py": "python",
        }
        seen: set[str] = set()
        result: list[str] = []
        for skill in skills:
            s = skill.lower().strip()
            s = aliases.get(s, s)
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result

    # ── Skill experience estimation ───────────────────────────────────────────

    def _estimate_skill_experience(self, cv: ParsedCV) -> dict[str, int]:
        """
        Map each skill to the sum of months in jobs where it was mentioned.
        Falls back to total experience if no explicit job mapping exists.
        """
        skill_months: dict[str, int] = {}
        total_months = int(cv.total_experience_years * 12)

        for exp in cv.work_experience:
            for skill in exp.skills_mentioned:
                skill_lower = skill.lower()
                skill_months[skill_lower] = (
                    skill_months.get(skill_lower, 0) + exp.duration_months
                )

        # For skills not mapped to any job, assign proportional estimate
        for skill in cv.skills:
            sl = skill.lower()
            if sl not in skill_months and total_months > 0:
                skill_months[sl] = total_months // 2  # conservative estimate

        return skill_months

    # ── Seniority ─────────────────────────────────────────────────────────────

    def _infer_seniority(self, cv: ParsedCV) -> str:
        # 1. Check job titles in work experience
        all_titles = " ".join(
            e.title.lower() for e in cv.work_experience
        )
        for level, keywords in reversed(list(_SENIORITY_MAP.items())):
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", all_titles):
                    return level

        # 2. Fallback: years of experience heuristic
        yoe = cv.total_experience_years
        if yoe < 2:
            return "junior"
        elif yoe < 5:
            return "mid"
        elif yoe < 10:
            return "senior"
        else:
            return "lead"

    # ── Domain ────────────────────────────────────────────────────────────────

    def _infer_domain(self, skills: list[str]) -> str:
        skill_set = set(skills)
        best, best_count = "general", 0
        for domain, keywords in _DOMAIN_VOCAB.items():
            overlap = len(skill_set & keywords)
            if overlap > best_count:
                best, best_count = domain, overlap
        return best
