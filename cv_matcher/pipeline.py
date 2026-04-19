"""
XAI CV Matcher — Orchestration Pipeline

Wires all 6 layers together into a single callable:

    result = Pipeline().run(cv_source, job_source)

Each layer is independently replaceable — swap any class for your own
implementation as long as it satisfies the same interface contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from cv_matcher.extractors import CVFeatureExtractor, JobFeatureExtractor
from cv_matcher.explainability import XAIEngine
from cv_matcher.matching import SkillMatcher
from cv_matcher.models import MatchReport, ParsedCV, ParsedJob
from cv_matcher.parsers import CVParser, JobParser
from cv_matcher.reporting import ReportGenerator


class Pipeline:
    """
    End-to-end CV matching pipeline.

    Constructor accepts optional custom layer implementations:

        pipeline = Pipeline(
            cv_parser=MyCustomCVParser(),
            skill_matcher=MyFinetuedMatcher(),
        )
    """

    def __init__(
        self,
        cv_parser: CVParser | None = None,
        job_parser: JobParser | None = None,
        cv_extractor: CVFeatureExtractor | None = None,
        job_extractor: JobFeatureExtractor | None = None,
        skill_matcher: SkillMatcher | None = None,
        xai_engine: XAIEngine | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        self.cv_parser = cv_parser or CVParser()
        self.job_parser = job_parser or JobParser()
        self.cv_extractor = cv_extractor or CVFeatureExtractor()
        self.job_extractor = job_extractor or JobFeatureExtractor()
        self.skill_matcher = skill_matcher or SkillMatcher()
        self.xai_engine = xai_engine or XAIEngine()
        self.report_generator = report_generator or ReportGenerator()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        cv_source: Union[str, Path, bytes, dict],
        job_source: Union[str, Path, bytes, dict],
    ) -> MatchReport:
        """
        Run the full pipeline.

        Parameters
        ----------
        cv_source  : file path (PDF/DOCX/TXT), raw bytes, or plain-text string
        job_source : file path, plain-text string, or structured dict

        Returns
        -------
        MatchReport  — fully populated Pydantic model (JSON-serializable)
        """
        # Layer 1+2: Parse
        cv: ParsedCV = self.cv_parser.parse(cv_source)
        job: ParsedJob = self.job_parser.parse(job_source)

        # Layer 3: Extract features
        cv_features = self.cv_extractor.extract(cv)
        job_features = self.job_extractor.extract(job)

        # Layer 4: Match
        match_result = self.skill_matcher.match(cv_features, job_features)

        # Layer 5: Explain
        xai_report = self.xai_engine.explain(match_result, cv_features, job_features)

        # Layer 6: Build report
        report = self.report_generator.build(
            cv, job, cv_features, job_features, match_result, xai_report
        )

        # Store intermediate objects on the report for access by callers
        report.__dict__["_cv"] = cv
        report.__dict__["_job"] = job
        report.__dict__["_cv_features"] = cv_features
        report.__dict__["_job_features"] = job_features
        report.__dict__["_match_result"] = match_result
        report.__dict__["_xai_report"] = xai_report

        return report

    def run_and_print(
        self,
        cv_source: Union[str, Path, bytes, dict],
        job_source: Union[str, Path, bytes, dict],
    ) -> MatchReport:
        """Run the pipeline and immediately print the Rich terminal report."""
        report = self.run(cv_source, job_source)
        cv_features = report.__dict__.get("_cv_features")
        job_features = report.__dict__.get("_job_features")
        self.report_generator.print_rich(report, cv_features, job_features)
        return report
