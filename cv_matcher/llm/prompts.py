"""
Centralized prompt library for the XAI CV Matcher.

Design principles:
- System prompts are cache-friendly (static, placed first).
- User prompts are dynamic and injected at call time.
- Every prompt has a clear role: parse | enrich | narrate | advise.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Shared system prompt  (cached — never changes between calls)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert senior HR analyst and technical recruiter with 15+ years of \
experience evaluating candidates for software engineering, data science, and DevOps roles.

Your responsibilities:
- Interpret CV and job-description data with precision and fairness.
- Produce structured JSON when asked; produce fluent prose narratives when asked.
- Be concise, professional, and actionable — avoid filler phrases.
- Never hallucinate skills or experience that are not present in the provided data.
- When writing for candidates, be encouraging but honest about gaps.
- When writing for recruiters, be objective and highlight decision-relevant facts.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CV enrichment  (parse → structured JSON)
# ─────────────────────────────────────────────────────────────────────────────

CV_PARSE_SYSTEM = SYSTEM_PROMPT + """
You will receive raw CV/resume text.  Extract all available information and return \
ONLY a valid JSON object — no markdown fences, no commentary.
"""

CV_PARSE_USER = """\
Parse the following CV and return a JSON object with this exact schema:

{{
  "name": "string or null",
  "email": "string or null",
  "phone": "string or null",
  "linkedin": "string or null",
  "seniority_level": "junior | mid | senior | lead",
  "total_experience_years": float,
  "skills": ["list", "of", "technical", "skills"],
  "languages": ["list", "of", "spoken", "languages"],
  "education": [
    {{"degree": "string", "field": "string", "institution": "string", "year": int_or_null}}
  ],
  "certifications": ["list", "of", "certifications"],
  "summary": "2-3 sentence professional summary inferred from the CV"
}}

Rules:
- Extract only skills explicitly mentioned or clearly implied by project/role descriptions.
- Infer seniority from titles, years of experience, and scope of responsibilities.
- If a field cannot be determined, use null (not an empty string).

CV TEXT:
---
{cv_text}
---
"""

# ─────────────────────────────────────────────────────────────────────────────
# Job description enrichment  (parse → structured JSON)
# ─────────────────────────────────────────────────────────────────────────────

JOB_PARSE_SYSTEM = SYSTEM_PROMPT + """
You will receive a job description.  Extract all requirements and return \
ONLY a valid JSON object — no markdown fences, no commentary.
"""

JOB_PARSE_USER = """\
Parse the following job description and return a JSON object with this exact schema:

{{
  "title": "string",
  "company": "string or null",
  "domain": "frontend | backend | fullstack | data | ml_ai | devops | general",
  "experience_level": "junior | mid | senior | lead",
  "required_experience_years": float_or_null,
  "required_skills": ["list", "of", "must-have", "skills"],
  "nice_to_have": ["list", "of", "preferred", "but", "optional", "skills"],
  "responsibilities": ["bullet-point list of key duties"],
  "tech_stack": ["inferred primary tech stack"],
  "remote_friendly": true_or_false_or_null
}}

Rules:
- Distinguish clearly between required ("must have", "required") and nice-to-have ("preferred", "bonus").
- Normalise skill names to lowercase (e.g. "React.js" → "react").
- required_experience_years should be the minimum years asked for (not a range).

JOB DESCRIPTION:
---
{job_text}
---
"""

# ─────────────────────────────────────────────────────────────────────────────
# Recruiter narrative
# ─────────────────────────────────────────────────────────────────────────────

RECRUITER_NARRATIVE_USER = """\
Write a professional recruiter assessment for the following match result.

MATCH DATA:
- Overall score: {score}/100
- Skills matched: {matched_count}/{total_required}
- Matched skills: {matched_skills}
- Missing skills: {missing_skills}
- Candidate experience: {candidate_years} years  |  Required: {required_years}
- Seniority: candidate is {cv_seniority}, role expects {job_seniority}
- Domain alignment: {domain_match}
- Vocabulary-mismatch flags: {bias_count} detected

Write 3–5 concise sentences covering:
1. Overall hiring signal (strong / moderate / weak match)
2. Key strengths the candidate brings
3. Notable skill or experience gaps
4. A clear, specific hiring recommendation

Tone: professional, objective, decision-focused.  No bullet points — flowing prose only.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Candidate narrative
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATE_NARRATIVE_USER = """\
Write personalised feedback for a job applicant based on the following match result.

MATCH DATA:
- Score: {score}/100  ({matched_count} of {total_required} required skills matched)
- Your strengths for this role: {matched_skills}
- Skills to develop: {missing_skills}
- Your experience: {candidate_years} years  |  Role requires: {required_years}
- Seniority: you are {cv_seniority}, role targets {job_seniority}
- Top upskilling tip: {top_upskill_tip}

Write 4–6 sentences that:
1. Give an honest but encouraging overall assessment
2. Highlight 2–3 genuine strengths clearly aligned to this role
3. Identify the most critical gap and why it matters for this position
4. Suggest one concrete next action to improve their candidacy

Tone: warm, honest, motivating.  Speak directly to the candidate ("you", "your").  \
No bullet points — flowing prose only.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Personalised upskilling roadmap
# ─────────────────────────────────────────────────────────────────────────────

UPSKILLING_ROADMAP_USER = """\
Create a personalised learning roadmap for a candidate targeting the following role.

CANDIDATE PROFILE:
- Current skills: {current_skills}
- Experience: {candidate_years} years as {cv_seniority}
- Domain: {cv_domain}

TARGET ROLE:
- Title: {job_title}
- Required skills they lack: {missing_skills}
- Role seniority: {job_seniority}

Return ONLY a JSON array (no markdown, no commentary) with up to 5 items, each with:
{{
  "skill": "skill name",
  "priority": "critical | important | nice-to-have",
  "estimated_weeks": integer (realistic time to reach working proficiency),
  "resource": "specific course, book, or project recommendation",
  "why": "one sentence explaining why this skill matters for the target role"
}}

Order by priority (critical first), then by estimated learning time (shortest first).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Skill gap explanation  (single-skill deep-dive)
# ─────────────────────────────────────────────────────────────────────────────

SKILL_GAP_EXPLANATION_USER = """\
The candidate is missing the skill "{skill}" required for a {job_title} role.

Candidate background: {candidate_years} years in {cv_domain}, skills include {current_skills}.

In 2 sentences max:
1. Explain why "{skill}" is important for this specific role.
2. Suggest the single fastest path to demonstrate competency in it.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Bias / vocabulary-mismatch explanation
# ─────────────────────────────────────────────────────────────────────────────

BIAS_EXPLANATION_USER = """\
The automated CV matcher flagged a potential vocabulary mismatch:

- CV mentions: "{cv_skill}"
- Job requires: "{job_skill}"
- Semantic similarity: {similarity}%

In 1–2 sentences, explain to a non-technical recruiter:
1. Whether these skills are likely equivalent or just related.
2. What manual check they should do to resolve this flag.
"""
