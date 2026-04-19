"""
CLI entry point — `cv-match` command.

Usage examples:
    cv-match cv.pdf job.txt
    cv-match cv.docx job.txt --output report.json
    cv-match cv.txt job.txt --output report.md
    cv-match --cv "resume text..." --job "job desc..." --json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from cv_matcher.pipeline import Pipeline
from cv_matcher.reporting import ReportGenerator


@click.command()
@click.argument("cv_file", required=False, type=click.Path(exists=False))
@click.argument("job_file", required=False, type=click.Path(exists=False))
@click.option("--cv", "cv_text", default=None, help="Raw CV text (alternative to CV_FILE)")
@click.option("--job", "job_text", default=None, help="Raw job description text (alternative to JOB_FILE)")
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path (.json for JSON, .md for Markdown). Default: terminal.",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON report to stdout.")
@click.option("--markdown", "as_md", is_flag=True, help="Print Markdown report to stdout.")
@click.option("--quiet", is_flag=True, help="Suppress Rich terminal output.")
def main(
    cv_file: str | None,
    job_file: str | None,
    cv_text: str | None,
    job_text: str | None,
    output: str | None,
    as_json: bool,
    as_md: bool,
    quiet: bool,
) -> None:
    """
    XAI CV–Job Skill Matching System

    \b
    Accepts:
      CV_FILE   — Path to resume (.pdf / .docx / .txt)
      JOB_FILE  — Path to job description (.txt / .json)

    \b
    Or pass raw text via --cv and --job flags.
    """
    # Resolve CV source
    cv_source: str | Path = _resolve_source(cv_file, cv_text, "CV")
    job_source: str | Path = _resolve_source(job_file, job_text, "Job")

    pipeline = Pipeline()
    rg = ReportGenerator()

    try:
        report = pipeline.run(cv_source, job_source)
    except Exception as exc:
        click.echo(f"[ERROR] Pipeline failed: {exc}", err=True)
        raise SystemExit(1) from exc

    cv_features = report.__dict__.get("_cv_features")
    job_features = report.__dict__.get("_job_features")

    # ── JSON output ────────────────────────────────────────────────────────
    if as_json or (output and output.endswith(".json")):
        json_str = rg.to_json(report)
        if output:
            Path(output).write_text(json_str, encoding="utf-8")
            click.echo(f"JSON report saved → {output}")
        else:
            click.echo(json_str)
        return

    # ── Markdown output ────────────────────────────────────────────────────
    if as_md or (output and output.endswith(".md")):
        md_str = rg.to_markdown(report, cv_features, job_features)
        if output:
            Path(output).write_text(md_str, encoding="utf-8")
            click.echo(f"Markdown report saved → {output}")
        else:
            click.echo(md_str)
        return

    # ── Default: rich terminal ────────────────────────────────────────────
    if not quiet:
        rg.print_rich(report, cv_features, job_features)

    # Also write file if --output given (no extension → JSON)
    if output:
        ext = Path(output).suffix.lower()
        if ext == ".md":
            Path(output).write_text(
                rg.to_markdown(report, cv_features, job_features), encoding="utf-8"
            )
        else:
            Path(output).write_text(rg.to_json(report), encoding="utf-8")
        click.echo(f"\nReport saved → {output}")


def _resolve_source(file_arg: str | None, text_arg: str | None, label: str) -> str | Path:
    if file_arg:
        p = Path(file_arg)
        if p.exists():
            return p
        # Treat as raw text if file not found
        click.echo(
            f"[WARNING] {label} file '{file_arg}' not found — treating as raw text.",
            err=True,
        )
        return file_arg
    if text_arg:
        return text_arg
    click.echo(
        f"[ERROR] No {label} source provided. Pass a file path or --{label.lower()} text.",
        err=True,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
