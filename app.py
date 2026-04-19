"""
XAI CV Matcher — Gradio Web UI (Orange design system)

Run:
    .venv/bin/python app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import gradio as gr

import cv_matcher.llm.client as _llm_mod
_llm_mod._default_client = None

from cv_matcher.pipeline import Pipeline
from cv_matcher.reporting import ReportGenerator

_pipeline = Pipeline()
_rg = ReportGenerator()

# ── Orange design system CSS ──────────────────────────────────────────────────
ORANGE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
    background: #f5f5f5 !important;
    color: #1a1a1a !important;
    margin: 0 !important;
}

/* ── Hide default gradio header clutter ── */
.gradio-container > .main > .wrap > .padding { padding: 0 !important; }
footer { display: none !important; }

/* ── Top navbar ── */
#orange-nav {
    background: #000;
    padding: 0 32px;
    height: 56px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 100;
    width: 100%;
}

#orange-nav .nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
}

#orange-nav .nav-title {
    color: white;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ── Hero section ── */
#orange-hero {
    background: #000;
    color: white;
    padding: 48px 32px 40px;
    border-bottom: 4px solid #ff7900;
}

#orange-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 10px;
    color: white;
}

#orange-hero p {
    font-size: 1rem;
    color: #aaa;
    margin: 0;
    max-width: 600px;
}

/* ── Main layout wrapper ── */
#orange-main {
    max-width: 1280px;
    margin: 0 auto;
    padding: 32px 24px;
}

/* ── Cards ── */
.orange-card {
    background: white;
    border-radius: 4px;
    padding: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 0;
}

.orange-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #ff7900;
    margin-bottom: 16px;
}

/* ── Gradio component overrides ── */
.gradio-container .gr-panel,
.gradio-container .panel {
    background: white !important;
    border: none !important;
    border-radius: 4px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

/* Labels */
.gradio-container label span,
.gradio-container .label-wrap span {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: #555 !important;
    margin-bottom: 8px !important;
}

/* Inputs & textareas */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border: 1.5px solid #e0e0e0 !important;
    border-radius: 4px !important;
    font-family: inherit !important;
    font-size: 0.93rem !important;
    transition: border-color 0.2s !important;
    background: #fafafa !important;
    color: #1a1a1a !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #ff7900 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,121,0,0.12) !important;
    background: white !important;
}

/* File upload */
.gradio-container .upload-container,
.gradio-container .file-preview {
    border: 2px dashed #e0e0e0 !important;
    border-radius: 4px !important;
    background: #fafafa !important;
    transition: border-color 0.2s !important;
}

.gradio-container .upload-container:hover {
    border-color: #ff7900 !important;
    background: #fff9f4 !important;
}

/* Primary button — Orange style */
.gradio-container button.primary,
.gradio-container button[variant="primary"],
.gradio-container .gr-button-primary {
    background: #ff7900 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100% !important;
}

.gradio-container button.primary:hover,
.gradio-container button[variant="primary"]:hover {
    background: #e56e00 !important;
    transform: translateY(-1px) !important;
}

/* Secondary buttons */
.gradio-container button.secondary {
    background: white !important;
    color: #333 !important;
    border: 1.5px solid #e0e0e0 !important;
    border-radius: 4px !important;
}

/* Tabs */
.gradio-container .tab-nav {
    border-bottom: 2px solid #f0f0f0 !important;
    background: transparent !important;
    gap: 0 !important;
}

.gradio-container .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    color: #888 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    margin-bottom: -2px !important;
    transition: color 0.2s, border-color 0.2s !important;
}

.gradio-container .tab-nav button.selected {
    color: #1a1a1a !important;
    border-bottom-color: #ff7900 !important;
    background: transparent !important;
}

.gradio-container .tab-nav button:hover {
    color: #333 !important;
}

/* Markdown output */
.gradio-container .prose,
.gradio-container .markdown-body {
    font-size: 0.93rem !important;
    line-height: 1.7 !important;
    color: #333 !important;
}

.gradio-container .prose h2 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    border-bottom: 2px solid #ff7900 !important;
    padding-bottom: 6px !important;
    margin-top: 20px !important;
}

.gradio-container .prose h3 {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #555 !important;
    margin-top: 18px !important;
}

.gradio-container .prose table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 14px 0 !important;
}

.gradio-container .prose th {
    background: #000 !important;
    color: white !important;
    padding: 10px 14px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    text-align: left !important;
}

.gradio-container .prose td {
    padding: 10px 14px !important;
    border-bottom: 1px solid #f0f0f0 !important;
    font-size: 0.88rem !important;
}

.gradio-container .prose code {
    background: #fff3e0 !important;
    color: #ff7900 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* Examples section */
.gradio-container .examples-holder {
    border: 1.5px solid #f0f0f0 !important;
    border-radius: 4px !important;
    margin-top: 12px !important;
}

/* Row gap */
.gradio-container .gap { gap: 20px !important; }

/* Score output area */
#score-panel .prose h2:first-child {
    font-size: 1.6rem !important;
    border: none !important;
    color: #ff7900 !important;
}
"""

# ── Score builder ─────────────────────────────────────────────────────────────

def _build_score_md(report) -> str:
    matched = len(report.matched_skills)
    total   = matched + len(report.missing_skills)
    pct     = report.overall_score

    bar_filled  = int(pct / 100 * 20)
    bar         = "█" * bar_filled + "░" * (20 - bar_filled)
    label       = "Strong match" if pct >= 75 else "Moderate match" if pct >= 55 else "Weak match"

    md = f"""## {pct:.0f} / 100 — {label}
`{bar} {pct:.0f}%`

| Dimension | Score |
|-----------|-------|
| Skill Match | {report.skill_score:.0f} / 100 |
| Experience | {report.experience_score:.0f} / 100 |
| Seniority | {"✅ Match" if report.seniority_match else "❌ Mismatch"} |
| Skills matched | {matched} / {total} |

### Matched Skills
{', '.join(f'`{m.job_skill}`' for m in report.matched_skills) or '_None_'}

### Missing Skills
{', '.join(f'`{g.missing_skill}`' for g in report.missing_skills) or '_None — great fit!_'}
"""
    if report.bias_flags:
        md += f"\n### ⚠ Vocabulary Flags ({len(report.bias_flags)})\n"
        for b in report.bias_flags:
            md += f"- `{b.cv_skill}` ↔ `{b.job_skill}` ({b.similarity*100:.0f}% similarity)\n"
    return md


# ── Main handler ──────────────────────────────────────────────────────────────

def match(cv_file, job_text: str):
    if cv_file is None:
        return "Please upload a CV file.", "", "", ""
    if not job_text.strip():
        return "Please paste a job description.", "", "", ""
    try:
        report       = _pipeline.run(cv_file.name, job_text)
        cv_features  = report.__dict__.get("_cv_features")
        job_features = report.__dict__.get("_job_features")
        return (
            _build_score_md(report),
            report.narrative_recruiter,
            report.narrative_candidate,
            _rg.to_markdown(report, cv_features, job_features),
        )
    except Exception as exc:
        return f"Error: {exc}", "", "", ""


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Orange CV Matcher") as demo:

    # Top navbar
    gr.HTML("""
    <div id="orange-nav">
      <div class="nav-logo">
        <svg width="28" height="28" viewBox="0 0 40 40">
          <circle cx="20" cy="20" r="20" fill="#ff7900"/>
          <circle cx="20" cy="20" r="10" fill="black"/>
        </svg>
        <span class="nav-title">Orange Business</span>
      </div>
    </div>
    """)

    # Hero
    gr.HTML("""
    <div id="orange-hero">
      <h1>CV Match Analyser</h1>
      <p>Upload a resume and paste a job description to get an AI-powered match score,
         skill gap analysis, and personalised feedback.</p>
    </div>
    """)

    with gr.Row(elem_id="orange-main"):

        # ── Left panel — inputs ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=340):
            gr.HTML('<div class="orange-card-title">01 — Upload Resume</div>')
            cv_upload = gr.File(
                label="CV file (PDF, DOCX, TXT)",
                file_types=[".pdf", ".docx", ".txt"],
            )

            gr.HTML('<div class="orange-card-title" style="margin-top:24px;">02 — Job Description</div>')
            job_input = gr.Textbox(
                label="Paste the full job description",
                placeholder="Senior Software Engineer at Acme Corp…\n\nRequirements:\n- 5+ years Python\n- Docker, Kubernetes…",
                lines=16,
            )

            submit_btn = gr.Button("Analyse Match →", variant="primary", size="lg")

            gr.Examples(
                examples=[[None, Path("examples/sample_job.txt").read_text()]],
                inputs=[cv_upload, job_input],
                label="Load sample job",
            )

        # ── Right panel — results ────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="orange-card-title">03 — Results</div>')
            with gr.Tabs():
                with gr.Tab("Score & Skills"):
                    score_out = gr.Markdown(
                        value="Results will appear here after analysis.",
                        elem_id="score-panel",
                    )

                with gr.Tab("Recruiter Feedback"):
                    recruiter_out = gr.Textbox(
                        label="AI assessment for recruiters",
                        lines=12,
                        interactive=False,
                    )

                with gr.Tab("Candidate Feedback"):
                    candidate_out = gr.Textbox(
                        label="Personalised feedback for the applicant",
                        lines=12,
                        interactive=False,
                    )

                with gr.Tab("Full Report"):
                    full_out = gr.Markdown()

    submit_btn.click(
        fn=match,
        inputs=[cv_upload, job_input],
        outputs=[score_out, recruiter_out, candidate_out, full_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=ORANGE_CSS)
