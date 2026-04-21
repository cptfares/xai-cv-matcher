"""
FastAPI REST wrapper around the CV matching pipeline.

Start:  uvicorn api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from cv_matcher.pipeline import Pipeline

app = FastAPI(title="CV Matcher API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
async def match(
    cv_file: UploadFile = File(...),
    job_description: str = Form(...),
):
    allowed = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(cv_file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX and TXT files are allowed")

    cv_bytes = await cv_file.read()
    if not cv_bytes:
        raise HTTPException(status_code=400, detail="CV file is empty")

    try:
        report = get_pipeline().run(cv_bytes, job_description)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return report.model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
