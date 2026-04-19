"""
Standalone integration test — no pip installs required.

Mocks out spacy, pdfplumber, python-docx, and sentence-transformers so the
full pipeline logic can be exercised with only stdlib + the packages already
available (pydantic / scikit-learn / numpy are checked).

Run with:
    python tests/test_pipeline_standalone.py
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Make cv_matcher importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Stub heavy optional dependencies ─────────────────────────────────────────

def _stub_spacy():
    """Return a minimal spaCy stand-in."""
    spacy = types.ModuleType("spacy")

    class _Doc:
        def __init__(self): self.ents = []
    class _NLP:
        def __call__(self, text): return _Doc()

    spacy.load = lambda name: _NLP()
    sys.modules["spacy"] = spacy


def _stub_sentence_transformers():
    st = types.ModuleType("sentence_transformers")

    class _FakeEncoder:
        def encode(self, texts):
            # Pure stdlib: deterministic pseudo-random floats, no numpy needed
            import random
            rng = random.Random(42)
            import array
            result = []
            for _ in texts:
                vec = [rng.random() for _ in range(64)]
                result.append(vec)
            # Return as a list of lists; sklearn cosine_similarity accepts this
            return result

    st.SentenceTransformer = lambda *a, **kw: _FakeEncoder()
    sys.modules["sentence_transformers"] = st


def _stub_pdfplumber():
    pdf = types.ModuleType("pdfplumber")
    sys.modules["pdfplumber"] = pdf


def _stub_docx():
    docx = types.ModuleType("docx")
    class _Doc:
        paragraphs = []
    docx.Document = lambda *a, **kw: _Doc()
    sys.modules["docx"] = docx


def _stub_shap():
    shap = types.ModuleType("shap")
    sys.modules["shap"] = shap


def _stub_rich():
    rich = types.ModuleType("rich")
    console_mod = types.ModuleType("rich.console")
    panel_mod = types.ModuleType("rich.panel")
    table_mod = types.ModuleType("rich.table")
    box_mod = types.ModuleType("rich.box")
    text_mod = types.ModuleType("rich.text")

    class _Console:
        def print(self, *a, **kw): pass
    class _Panel:
        def __init__(self, *a, **kw): pass
    class _Table:
        def __init__(self, *a, **kw): pass
        def add_column(self, *a, **kw): pass
        def add_row(self, *a, **kw): pass
    box_mod.ROUNDED = None
    box_mod.SIMPLE = None
    console_mod.Console = _Console
    panel_mod.Panel = _Panel
    table_mod.Table = _Table

    rich.console = console_mod
    rich.panel = panel_mod
    rich.table = table_mod
    rich.box = box_mod
    rich.text = text_mod
    sys.modules.update({
        "rich": rich,
        "rich.console": console_mod,
        "rich.panel": panel_mod,
        "rich.table": table_mod,
        "rich.box": box_mod,
        "rich.text": text_mod,
    })


# Install stubs before any cv_matcher import
_stub_spacy()
_stub_sentence_transformers()
_stub_pdfplumber()
_stub_docx()
_stub_shap()
_stub_rich()


# ── Now import the pipeline ───────────────────────────────────────────────────
from cv_matcher.pipeline import Pipeline  # noqa: E402
from cv_matcher.reporting import ReportGenerator  # noqa: E402


# ── Test data ─────────────────────────────────────────────────────────────────
SAMPLE_CV = """
Sarah Johnson
sarah.johnson@email.com | +1-415-555-0192

SKILLS
Python, TypeScript, React, Node.js, PostgreSQL, Docker, AWS, CI/CD, GraphQL

WORK EXPERIENCE
Senior Software Engineer — TechCorp Inc.
Jan 2021 – Present
Built React dashboards and FastAPI microservices deployed on Docker and AWS.
Skills: Python, React, Docker, AWS, FastAPI, PostgreSQL

Software Engineer — DataFlow Ltd.
Mar 2018 – Dec 2020
Developed ETL pipelines using Python and maintained CI/CD on GitHub Actions.
Skills: Python, CI/CD, GitHub Actions

EDUCATION
BSc Computer Science, UC Berkeley, 2017

CERTIFICATIONS
AWS Solutions Architect Associate
"""

SAMPLE_JOB = {
    "title": "Senior Full-Stack Engineer",
    "company": "StartupX",
    "required_skills": [
        "typescript", "react", "node.js", "postgresql",
        "docker", "kubernetes", "aws", "ci/cd", "graphql",
    ],
    "nice_to_have": ["terraform", "kafka"],
    "experience_level": "senior",
    "required_experience_years": 5,
    "responsibilities": [
        "Lead frontend and backend feature development",
        "Set up CI/CD pipelines",
        "Conduct code reviews",
    ],
}


class TestPipelineEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = Pipeline()
        cls.rg = ReportGenerator()
        cls.report = cls.pipeline.run(SAMPLE_CV, SAMPLE_JOB)

    def test_report_has_overall_score(self):
        self.assertIsInstance(self.report.overall_score, float)
        self.assertGreaterEqual(self.report.overall_score, 0)
        self.assertLessEqual(self.report.overall_score, 100)

    def test_report_has_matched_skills(self):
        self.assertIsInstance(self.report.matched_skills, list)

    def test_report_has_missing_skills(self):
        self.assertIsInstance(self.report.missing_skills, list)

    def test_report_has_feature_importances(self):
        self.assertIsInstance(self.report.feature_importances, list)
        if self.report.feature_importances:
            fi = self.report.feature_importances[0]
            self.assertIn(fi.direction, ("positive", "negative", "neutral"))

    def test_report_has_narratives(self):
        self.assertIsInstance(self.report.narrative_recruiter, str)
        self.assertIsInstance(self.report.narrative_candidate, str)
        self.assertGreater(len(self.report.narrative_recruiter), 20)
        self.assertGreater(len(self.report.narrative_candidate), 20)

    def test_json_serializable(self):
        import json
        json_str = self.rg.to_json(self.report)
        data = json.loads(json_str)
        self.assertIn("overall_score", data)
        self.assertIn("matched_skills", data)
        self.assertIn("missing_skills", data)
        self.assertIn("feature_importances", data)

    def test_markdown_output(self):
        cv_features = self.report.__dict__.get("_cv_features")
        job_features = self.report.__dict__.get("_job_features")
        md = self.rg.to_markdown(self.report, cv_features, job_features)
        self.assertIn("# CV Match Report", md)
        self.assertIn("Overall Score", md)
        self.assertIn("Matched Skills", md)
        self.assertIn("XAI", md)

    def test_cv_name_extracted(self):
        # Name detection is regex/NLP — just check it's a non-empty string
        self.assertIsInstance(self.report.cv_name, str)

    def test_job_title_extracted(self):
        self.assertEqual(self.report.job_title, "Senior Full-Stack Engineer")

    def test_skill_weights_sum_to_one(self):
        job_features = self.report.__dict__.get("_job_features")
        if job_features and job_features.skill_weights:
            total = sum(job_features.skill_weights.values())
            self.assertAlmostEqual(total, 1.0, places=2)

    def test_bias_flags_structure(self):
        for flag in self.report.bias_flags:
            self.assertIsInstance(flag.cv_skill, str)
            self.assertIsInstance(flag.job_skill, str)
            self.assertGreater(flag.similarity, 0)


class TestParsers(unittest.TestCase):

    def test_cv_parser_extracts_email(self):
        from cv_matcher.parsers import CVParser
        cv = CVParser().parse("John Doe\njohn@example.com\n\nSKILLS\npython, docker")
        self.assertEqual(cv.email, "john@example.com")

    def test_cv_parser_extracts_skills(self):
        from cv_matcher.parsers import CVParser
        cv = CVParser().parse("SKILLS\npython, docker, react, postgresql")
        self.assertIn("python", cv.skills)
        self.assertIn("docker", cv.skills)

    def test_job_parser_dict_mode(self):
        from cv_matcher.parsers import JobParser
        job = JobParser().parse({
            "title": "Data Engineer",
            "required_skills": ["python", "spark", "sql"],
            "experience_level": "mid",
        })
        self.assertEqual(job.title, "Data Engineer")
        skill_names = [r.skill for r in job.requirements]
        for s in ["python", "spark", "sql"]:
            self.assertIn(s, skill_names)

    def test_job_parser_text_mode(self):
        from cv_matcher.parsers import JobParser
        text = "Machine Learning Engineer\n\nRequirements\npython, pytorch, scikit-learn\n3+ years of experience"
        job = JobParser().parse(text)
        self.assertIsInstance(job.requirements, list)


class TestFeatureExtractors(unittest.TestCase):

    def test_cv_feature_normalization(self):
        from cv_matcher.extractors import CVFeatureExtractor
        from cv_matcher.models import ParsedCV
        cv = ParsedCV(skills=["React.js", "NodeJS", "Postgres", "K8s"])
        feats = CVFeatureExtractor().extract(cv)
        self.assertIn("react", feats.normalized_skills)
        self.assertIn("node.js", feats.normalized_skills)
        self.assertIn("postgresql", feats.normalized_skills)
        self.assertIn("kubernetes", feats.normalized_skills)

    def test_seniority_from_years(self):
        from cv_matcher.extractors import CVFeatureExtractor
        from cv_matcher.models import ParsedCV
        cv = ParsedCV(total_experience_years=7)
        feats = CVFeatureExtractor().extract(cv)
        self.assertEqual(feats.seniority_level, "senior")


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)
