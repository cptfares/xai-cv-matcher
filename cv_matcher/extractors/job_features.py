"""
Layer 3 — Job Feature Extractor

Normalizes parsed job data into structured features.
"""
from __future__ import annotations

from cv_matcher.models import JobFeatures, ParsedJob


class JobFeatureExtractor:
    """Convert a ParsedJob into a JobFeatures object."""

    _ALIASES: dict[str, str] = {
        "react.js": "react",
        "reactjs": "react",
        "node": "node.js",
        "nodejs": "node.js",
        "postgres": "postgresql",
        "k8s": "kubernetes",
        "vue.js": "vue",
        "vuejs": "vue",
    }

    def extract(self, job: ParsedJob) -> JobFeatures:
        required_skills: list[str] = []
        skill_weights: dict[str, float] = {}

        for req in job.requirements:
            normalized = self._normalize(req.skill)
            if normalized not in required_skills:
                required_skills.append(normalized)
                skill_weights[normalized] = req.weight

        # Normalize weights to sum to 1 (for clean contribution math)
        total_w = sum(skill_weights.values()) or 1.0
        skill_weights = {k: round(v / total_w, 4) for k, v in skill_weights.items()}

        return JobFeatures(
            required_skills=required_skills,
            skill_weights=skill_weights,
            experience_level=job.experience_level,
            required_experience_years=job.required_experience_years,
            domain=job.domain,
        )

    def _normalize(self, skill: str) -> str:
        s = skill.lower().strip()
        return self._ALIASES.get(s, s)
