"""
Shared Pydantic models for all pipeline layers.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Layer 2: Parsed documents ────────────────────────────────────────────────

class WorkExperience(BaseModel):
    title: str = ""
    company: str = ""
    duration_months: int = 0
    description: str = ""
    skills_mentioned: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: str = ""
    field: str = ""
    institution: str = ""
    year: Optional[int] = None


class ParsedCV(BaseModel):
    raw_text: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    skills: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    total_experience_years: float = 0.0


class JobRequirement(BaseModel):
    skill: str
    weight: float = 1.0          # derived from frequency / position in JD
    is_required: bool = True     # False → nice-to-have


class ParsedJob(BaseModel):
    raw_text: str = ""
    title: str = ""
    company: str = ""
    experience_level: str = ""   # junior / mid / senior / lead
    required_experience_years: Optional[float] = None
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[JobRequirement] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    domain: str = ""


# ── Layer 3: Extracted features ──────────────────────────────────────────────

class CVFeatures(BaseModel):
    normalized_skills: list[str] = Field(default_factory=list)
    skill_experience_months: dict[str, int] = Field(default_factory=dict)
    seniority_level: str = ""    # junior / mid / senior / lead
    domain: str = ""
    total_experience_years: float = 0.0


class JobFeatures(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    skill_weights: dict[str, float] = Field(default_factory=dict)
    experience_level: str = ""
    required_experience_years: Optional[float] = None
    domain: str = ""


# ── Layer 4: Skill match results ─────────────────────────────────────────────

class SkillMatch(BaseModel):
    job_skill: str
    cv_skill: Optional[str]         # best matching CV skill (None if missing)
    similarity_score: float         # 0.0–1.0
    weight: float                   # job-side importance weight
    weighted_contribution: float    # similarity_score × weight
    is_matched: bool                # above threshold
    is_semantic_synonym: bool = False  # matched via embedding, not exact string


class MatchResult(BaseModel):
    overall_score: float            # 0–100
    skill_matches: list[SkillMatch] = Field(default_factory=list)
    matched_count: int = 0
    missing_count: int = 0
    total_required: int = 0
    experience_score: float = 0.0   # 0–100, based on years comparison
    seniority_match: bool = False


# ── Layer 5: XAI output ──────────────────────────────────────────────────────

class SkillContribution(BaseModel):
    skill: str
    shap_value: float               # contribution to overall score delta
    direction: str                  # "positive" | "negative" | "neutral"
    explanation: str


class GapItem(BaseModel):
    missing_skill: str
    importance: float               # derived from job weight
    upskilling_suggestion: str = ""


class BiasFlag(BaseModel):
    cv_skill: str
    job_skill: str
    similarity: float
    note: str                       # e.g., "Semantic synonyms penalized by exact match"


class XAIReport(BaseModel):
    feature_importances: list[SkillContribution] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    bias_flags: list[BiasFlag] = Field(default_factory=list)
    narrative_recruiter: str = ""
    narrative_candidate: str = ""


# ── Layer 6: Final output ────────────────────────────────────────────────────

class MatchReport(BaseModel):
    # Metadata
    cv_name: str = ""
    job_title: str = ""
    generated_at: str = ""

    # Scores
    overall_score: float = 0.0      # 0–100
    skill_score: float = 0.0
    experience_score: float = 0.0
    seniority_match: bool = False

    # Matched & missing
    matched_skills: list[SkillMatch] = Field(default_factory=list)
    missing_skills: list[GapItem] = Field(default_factory=list)

    # XAI
    feature_importances: list[SkillContribution] = Field(default_factory=list)
    bias_flags: list[BiasFlag] = Field(default_factory=list)

    # Narratives
    narrative_recruiter: str = ""
    narrative_candidate: str = ""

    # Upskilling
    upskilling_suggestions: list[str] = Field(default_factory=list)
