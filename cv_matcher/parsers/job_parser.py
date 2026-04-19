"""
Layer 1 + 2 — Job Offer Parser

Accepts structured dict or free-text job description.
Extracts: title, required skills (with implicit weighting), experience level,
          responsibilities, nice-to-haves, domain.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Union

import spacy

from cv_matcher.models import JobRequirement, ParsedJob
from cv_matcher.parsers.cv_parser import _TECH_SKILLS_VOCAB, _nlp
from cv_matcher.llm.client import get_client
from cv_matcher.llm.prompts import JOB_PARSE_SYSTEM, JOB_PARSE_USER

_SENIORITY_MAP = {
    "junior":    re.compile(r"\bjunior|jr\.?\b|entry.?level|trainee\b", re.I),
    "mid":       re.compile(r"\bmid.?level|associate\b", re.I),
    "senior":    re.compile(r"\bsenior|sr\.?\b|principal\b", re.I),
    "lead":      re.compile(r"\blead|staff\b|head of|director|manager|architect\b", re.I),
}

_EXP_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:–|-|to)?\s*(\d+)?\s*years?\s+(?:of\s+)?(?:experience|exp)", re.I
)

_SECTION_PATTERNS = {
    "requirements":    re.compile(r"^requirements?|qualifications?|what you.?ll need|must.?have", re.I),
    "responsibilities": re.compile(r"^responsibilities|what you.?ll do|role|duties|day.to.day", re.I),
    "nice_to_have":    re.compile(r"nice.?to.?have|bonus|preferred|plus|desirable", re.I),
    "about":           re.compile(r"^about\s+(the\s+)?(role|position|company|us)", re.I),
}

_DOMAIN_KEYWORDS = {
    "frontend":    {"react", "vue", "angular", "html", "css", "ui", "ux"},
    "backend":     {"django", "flask", "fastapi", "spring", "node.js", "express"},
    "data":        {"pandas", "numpy", "sql", "spark", "hadoop", "etl", "dbt"},
    "ml_ai":       {"tensorflow", "pytorch", "scikit-learn", "mlflow", "llm", "hugging face"},
    "devops":      {"docker", "kubernetes", "terraform", "ansible", "ci/cd", "jenkins"},
    "fullstack":   {"react", "node.js", "postgresql", "docker"},
}


class JobParser:
    """Parse a job offer from plain text or a structured dict."""

    def parse(self, source: Union[str, Path, dict]) -> ParsedJob:
        if isinstance(source, dict):
            return self._parse_dict(source)
        path = Path(str(source))
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(source)
        return self._parse_text(text)

    # ── Dict mode (structured input) ──────────────────────────────────────────

    def _parse_dict(self, d: dict) -> ParsedJob:
        job = ParsedJob(
            raw_text=str(d),
            title=d.get("title", ""),
            company=d.get("company", ""),
        )
        all_skills: list[str] = d.get("required_skills", [])
        job.nice_to_have = d.get("nice_to_have", [])
        job.responsibilities = d.get("responsibilities", [])
        job.experience_level = d.get("experience_level", "")

        exp = d.get("required_experience_years")
        job.required_experience_years = float(exp) if exp else None

        job.requirements = self._build_weighted_requirements(
            all_skills, job.raw_text, required=True
        )
        # Nice-to-haves with lower base weight
        job.requirements += self._build_weighted_requirements(
            job.nice_to_have, "", required=False, base_weight=0.5
        )
        job.domain = self._infer_domain(all_skills + job.nice_to_have)
        if not job.experience_level:
            job.experience_level = self._infer_seniority(job.raw_text)
        return job

    # ── Free-text mode ────────────────────────────────────────────────────────

    def _parse_text(self, text: str) -> ParsedJob:
        job = ParsedJob(raw_text=text)

        # Title: first non-blank line or PERSON/ORG-like entity
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        job.title = lines[0] if lines else ""

        # Seniority
        job.experience_level = self._infer_seniority(text)

        # Required years
        job.required_experience_years = self._extract_years(text)

        # Section-based extraction
        sections = self._segment_sections(text)

        req_text = sections.get("requirements", "") + "\n" + sections.get("about", "")
        resp_text = sections.get("responsibilities", "")
        nice_text = sections.get("nice_to_have", "")

        # Skills from requirements section (higher weight)
        req_skills = self._extract_skills_from_text(req_text)
        nice_skills = self._extract_skills_from_text(nice_text)

        # Fallback: extract from full text if sections are empty
        if not req_skills:
            req_skills = self._extract_skills_from_text(text)

        job.requirements = self._build_weighted_requirements(
            req_skills, req_text, required=True
        )
        job.requirements += self._build_weighted_requirements(
            nice_skills, nice_text, required=False, base_weight=0.5
        )
        job.responsibilities = [
            l.strip(" ●•-–—") for l in resp_text.splitlines()
            if l.strip(" ●•-–—") and len(l.strip()) > 10
        ]
        job.nice_to_have = nice_skills
        job.domain = self._infer_domain(req_skills + nice_skills)

        return job

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _segment_sections(self, text: str) -> dict[str, str]:
        sections: dict[str, list[str]] = {"_default": []}
        current = "_default"
        for line in text.splitlines():
            stripped = line.strip()
            matched = self._classify_section(stripped)
            if matched and len(stripped) < 70:
                current = matched
                sections.setdefault(current, [])
            else:
                sections.setdefault(current, []).append(line)
        return {k: "\n".join(v) for k, v in sections.items()}

    def _classify_section(self, line: str) -> str | None:
        clean = line.strip(":–—●•").strip()
        for name, pat in _SECTION_PATTERNS.items():
            if pat.match(clean):
                return name
        return None

    def _extract_skills_from_text(self, text: str) -> list[str]:
        found: set[str] = set()
        lower = text.lower()
        for skill in _TECH_SKILLS_VOCAB:
            if re.search(r"\b" + re.escape(skill) + r"\b", lower):
                found.add(skill)
        # Also capture comma/bullet items
        for token in re.split(r"[,\n•\|/]", text):
            token = token.strip(" :-–—●•\t")
            if 2 <= len(token) <= 40 and token.lower() in _TECH_SKILLS_VOCAB:
                found.add(token.lower())
        return sorted(found)

    def _build_weighted_requirements(
        self,
        skills: list[str],
        source_text: str,
        required: bool = True,
        base_weight: float = 1.0,
    ) -> list[JobRequirement]:
        """
        Weight each skill by:
        - How often it appears in the source text (frequency bonus)
        - Its position (skills in first half of text get +0.1)
        """
        lower = source_text.lower()
        half = len(lower) // 2

        reqs: list[JobRequirement] = []
        for skill in skills:
            # Frequency count
            count = len(re.findall(r"\b" + re.escape(skill) + r"\b", lower))
            freq_bonus = min(count - 1, 3) * 0.1  # up to +0.3

            # Position bonus
            first_pos = lower.find(skill)
            position_bonus = 0.1 if (first_pos != -1 and first_pos < half) else 0.0

            weight = round(base_weight + freq_bonus + position_bonus, 2)
            reqs.append(
                JobRequirement(skill=skill, weight=weight, is_required=required)
            )

        # Normalize weights so max = 1.5 * base_weight
        if reqs:
            max_w = max(r.weight for r in reqs)
            cap = base_weight * 1.5
            if max_w > cap:
                factor = cap / max_w
                for r in reqs:
                    r.weight = round(r.weight * factor, 2)

        return reqs

    def _extract_years(self, text: str) -> float | None:
        m = _EXP_YEARS_RE.search(text)
        if m:
            low = float(m.group(1))
            high = float(m.group(2)) if m.group(2) else low
            return (low + high) / 2
        return None

    def _infer_seniority(self, text: str) -> str:
        for level, pat in _SENIORITY_MAP.items():
            if pat.search(text):
                return level
        return "mid"  # default assumption

    def _infer_domain(self, skills: list[str]) -> str:
        skill_set = set(s.lower() for s in skills)
        best, best_count = "general", 0
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            overlap = len(skill_set & keywords)
            if overlap > best_count:
                best, best_count = domain, overlap
        return best
