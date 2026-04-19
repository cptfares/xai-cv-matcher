"""
Quick demo — run from the project root:

    python examples/demo.py

No external models needed: falls back to TF-IDF if sentence-transformers
is not installed. With the full requirements installed you get semantic
embeddings and much richer matching.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Make sure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_matcher.pipeline import Pipeline
from cv_matcher.reporting import ReportGenerator

HERE = Path(__file__).parent


def main():
    pipeline = Pipeline()
    rg = ReportGenerator()

    cv_path = HERE / "sample_cv.txt"
    job_path = HERE / "sample_job.txt"

    print("Running XAI CV Matcher pipeline…\n")
    report = pipeline.run(cv_path, job_path)

    cv_features = report.__dict__.get("_cv_features")
    job_features = report.__dict__.get("_job_features")

    # ── Rich terminal output ──────────────────────────────────────────────
    rg.print_rich(report, cv_features, job_features)

    # ── Save Markdown report ──────────────────────────────────────────────
    md_path = HERE / "report_output.md"
    md_path.write_text(rg.to_markdown(report, cv_features, job_features))
    print(f"\nMarkdown report saved → {md_path}")

    # ── Save JSON report ──────────────────────────────────────────────────
    json_path = HERE / "report_output.json"
    json_path.write_text(rg.to_json(report))
    print(f"JSON report saved     → {json_path}")

    # ── Quick summary ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Overall Score : {report.overall_score:.0f} / 100")
    print(f"Matched Skills: {len(report.matched_skills)}")
    print(f"Missing Skills: {len(report.missing_skills)}")
    print(f"Bias Flags    : {len(report.bias_flags)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
