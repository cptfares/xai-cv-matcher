"""
Layer 6 — Report Generator

Combines MatchResult + XAIReport into:
  1. A structured MatchReport (Pydantic → JSON-serializable)
  2. A human-readable Rich terminal report
  3. A Markdown report string (for file export or API response)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TextIO

from cv_matcher.models import (
    CVFeatures,
    GapItem,
    JobFeatures,
    MatchReport,
    MatchResult,
    ParsedCV,
    ParsedJob,
    XAIReport,
)


class ReportGenerator:

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        cv: ParsedCV,
        job: ParsedJob,
        cv_features: CVFeatures,
        job_features: JobFeatures,
        match: MatchResult,
        xai: XAIReport,
    ) -> MatchReport:
        matched = [m for m in match.skill_matches if m.is_matched]
        upskilling = [
            f"{g.missing_skill}: {g.upskilling_suggestion}"
            for g in xai.gaps
            if g.upskilling_suggestion
        ]

        return MatchReport(
            cv_name=cv.name or "Unknown",
            job_title=job.title or "Unknown",
            generated_at=datetime.now(timezone.utc).isoformat(),
            overall_score=match.overall_score,
            skill_score=self._skill_score(match),
            experience_score=match.experience_score,
            seniority_match=match.seniority_match,
            matched_skills=matched,
            missing_skills=xai.gaps,
            feature_importances=xai.feature_importances,
            bias_flags=xai.bias_flags,
            narrative_recruiter=xai.narrative_recruiter,
            narrative_candidate=xai.narrative_candidate,
            upskilling_suggestions=upskilling,
        )

    def to_json(self, report: MatchReport, indent: int = 2) -> str:
        """Return a JSON string of the full report."""
        return json.dumps(report.model_dump(), indent=indent, ensure_ascii=False)

    def to_markdown(
        self,
        report: MatchReport,
        cv_features: CVFeatures,
        job_features: JobFeatures,
    ) -> str:
        """Return a Markdown string suitable for file export or API responses."""
        lines: list[str] = []
        a = lines.append

        a(f"# CV Match Report")
        a(f"")
        a(f"**Candidate:** {report.cv_name}  ")
        a(f"**Position:** {report.job_title}  ")
        a(f"**Generated:** {report.generated_at}")
        a(f"")

        # Score summary
        a(f"---")
        a(f"## Overall Score: {report.overall_score:.0f} / 100")
        a(f"")
        score_bar = self._score_bar(report.overall_score)
        a(f"`{score_bar}`")
        a(f"")
        a(f"| Dimension | Score |")
        a(f"|-----------|-------|")
        a(f"| Skill Match | {report.skill_score:.1f} / 100 |")
        a(f"| Experience | {report.experience_score:.1f} / 100 |")
        a(f"| Seniority | {'✓ Match' if report.seniority_match else '✗ Mismatch'} |")
        a(f"")

        # Matched skills
        a(f"---")
        a(f"## Matched Skills ({len(report.matched_skills)} of "
          f"{len(report.matched_skills) + len(report.missing_skills)})")
        a(f"")
        if report.matched_skills:
            a(f"| Job Skill | CV Skill | Similarity | Semantic? |")
            a(f"|-----------|----------|-----------|-----------|")
            for m in sorted(report.matched_skills, key=lambda x: -x.similarity_score):
                sem = "Yes" if m.is_semantic_synonym else "No"
                a(f"| {m.job_skill} | {m.cv_skill or '-'} | "
                  f"{m.similarity_score * 100:.0f}% | {sem} |")
        else:
            a(f"_No skills matched._")
        a(f"")

        # Missing skills
        a(f"---")
        a(f"## Missing / Weak Skills ({len(report.missing_skills)})")
        a(f"")
        if report.missing_skills:
            a(f"| Skill | Importance | Upskilling Suggestion |")
            a(f"|-------|------------|----------------------|")
            for g in report.missing_skills:
                imp_stars = "★" * round(g.importance * 5) + "☆" * (5 - round(g.importance * 5))
                a(f"| {g.missing_skill} | {imp_stars} | {g.upskilling_suggestion[:80]}… |")
        else:
            a(f"_All required skills are present._")
        a(f"")

        # XAI Feature importances
        a(f"---")
        a(f"## XAI — Feature Importances (SHAP-inspired)")
        a(f"")
        a(f"Contribution of each skill to the overall match score:")
        a(f"")
        if report.feature_importances:
            for fi in report.feature_importances[:10]:  # top 10
                bar = self._shap_bar(fi.shap_value)
                icon = "▲" if fi.direction == "positive" else ("▼" if fi.direction == "negative" else "━")
                a(f"- **{fi.skill}** {icon} `{bar}` ({fi.shap_value:+.3f})  ")
                a(f"  _{fi.explanation}_")
        a(f"")

        # Bias flags
        if report.bias_flags:
            a(f"---")
            a(f"## Bias Flags ({len(report.bias_flags)})")
            a(f"")
            a(f"> **Note:** The following skill pairs scored near the match threshold.")
            a(f"> The model may be penalizing synonymous vocabulary. Review manually.")
            a(f"")
            for bf in report.bias_flags:
                a(f"- **CV skill:** `{bf.cv_skill}` → **Job skill:** `{bf.job_skill}` "
                  f"(similarity: {bf.similarity * 100:.0f}%)")
                a(f"  _{bf.note}_")
            a(f"")

        # Narratives
        a(f"---")
        a(f"## For Recruiters")
        a(f"")
        a(f"{report.narrative_recruiter}")
        a(f"")
        a(f"---")
        a(f"## For Candidates")
        a(f"")
        a(f"{report.narrative_candidate}")
        a(f"")

        # Upskilling
        if report.upskilling_suggestions:
            a(f"---")
            a(f"## Upskilling Roadmap")
            a(f"")
            for sug in report.upskilling_suggestions:
                a(f"- {sug}")
            a(f"")

        return "\n".join(lines)

    def print_rich(self, report: MatchReport, cv_features: CVFeatures, job_features: JobFeatures) -> None:
        """Print a formatted report to the terminal using Rich."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich import box
            from rich.text import Text
        except ImportError:
            print(self.to_markdown(report, cv_features, job_features))
            return

        console = Console()

        # ── Header ────────────────────────────────────────────────────────
        console.print()
        console.print(Panel(
            f"[bold white]Candidate:[/] {report.cv_name}\n"
            f"[bold white]Position:[/]  {report.job_title}\n"
            f"[bold white]Generated:[/] {report.generated_at}",
            title="[bold cyan]XAI CV Match Report[/]",
            border_style="cyan",
        ))

        # ── Score banner ──────────────────────────────────────────────────
        score = report.overall_score
        color = "green" if score >= 75 else "yellow" if score >= 55 else "red"
        console.print(
            Panel(
                f"[bold {color}]{score:.0f} / 100[/]\n"
                f"[dim]{self._score_bar(score)}[/]",
                title="[bold]Overall Match Score[/]",
                border_style=color,
            )
        )

        # ── Sub-scores ────────────────────────────────────────────────────
        sub = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        sub.add_column("Dimension")
        sub.add_column("Score", justify="right")
        sub.add_row("Skill Match", f"{report.skill_score:.1f} / 100")
        sub.add_row("Experience", f"{report.experience_score:.1f} / 100")
        sub.add_row("Seniority", "✓ Match" if report.seniority_match else "✗ Mismatch")
        console.print(sub)
        console.print()

        # ── Matched skills ────────────────────────────────────────────────
        mtable = Table(
            title=f"Matched Skills ({len(report.matched_skills)})",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold green",
        )
        mtable.add_column("Job Skill")
        mtable.add_column("CV Skill")
        mtable.add_column("Similarity", justify="right")
        mtable.add_column("Semantic?")
        for m in sorted(report.matched_skills, key=lambda x: -x.similarity_score):
            mtable.add_row(
                m.job_skill,
                m.cv_skill or "-",
                f"{m.similarity_score * 100:.0f}%",
                "[yellow]yes[/]" if m.is_semantic_synonym else "no",
            )
        console.print(mtable)

        # ── Missing skills ────────────────────────────────────────────────
        if report.missing_skills:
            gtable = Table(
                title=f"Missing / Weak Skills ({len(report.missing_skills)})",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold red",
            )
            gtable.add_column("Skill")
            gtable.add_column("Importance", justify="right")
            gtable.add_column("Upskilling Path")
            for g in report.missing_skills:
                imp = "★" * round(g.importance * 5)
                suggestion = g.upskilling_suggestion[:60] + "…" if len(g.upskilling_suggestion) > 60 else g.upskilling_suggestion
                gtable.add_row(g.missing_skill, imp or "-", suggestion)
            console.print(gtable)

        # ── SHAP feature importances ──────────────────────────────────────
        console.print()
        console.print("[bold]XAI — Top Feature Importances (SHAP-inspired)[/]")
        for fi in report.feature_importances[:8]:
            color = "green" if fi.direction == "positive" else ("red" if fi.direction == "negative" else "dim")
            bar = self._shap_bar(fi.shap_value)
            console.print(
                f"  [{color}]{fi.shap_value:+.3f}[/] [bold]{fi.skill}[/]  [dim]{bar}[/]"
            )
            console.print(f"  [dim italic]{fi.explanation}[/]")

        # ── Bias flags ────────────────────────────────────────────────────
        if report.bias_flags:
            console.print()
            console.print(f"[bold yellow]⚠ Bias Flags ({len(report.bias_flags)})[/]")
            for bf in report.bias_flags:
                console.print(
                    f"  [yellow]'{bf.cv_skill}'[/] ↔ [yellow]'{bf.job_skill}'[/] "
                    f"({bf.similarity * 100:.0f}% similarity — below threshold)"
                )

        # ── Narratives ────────────────────────────────────────────────────
        console.print()
        console.print(Panel(report.narrative_recruiter, title="[bold]For Recruiters[/]", border_style="blue"))
        console.print(Panel(report.narrative_candidate, title="[bold]For Candidates[/]", border_style="green"))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _skill_score(self, match: MatchResult) -> float:
        """Reconstruct skill-only score (before experience/seniority blending)."""
        if not match.skill_matches:
            return 0.0
        total_w = sum(m.weight for m in match.skill_matches) or 1.0
        raw = sum(m.weighted_contribution for m in match.skill_matches)
        return min(100.0, round(raw / total_w * 100, 1))

    def _score_bar(self, score: float, width: int = 30) -> str:
        filled = int(score / 100 * width)
        return "█" * filled + "░" * (width - filled) + f" {score:.0f}%"

    def _shap_bar(self, shap: float, width: int = 12) -> str:
        mid = width // 2
        magnitude = int(abs(shap) * mid * 4)
        magnitude = min(magnitude, mid)
        if shap > 0:
            return "─" * mid + "▶" * magnitude + " " * (mid - magnitude)
        else:
            return " " * (mid - magnitude) + "◀" * magnitude + "─" * mid
