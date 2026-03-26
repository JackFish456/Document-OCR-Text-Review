"""Persist reviewer compare artifacts for HTTP download (TTL cleanup)."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.reporting.reviewer_bundle import (
    PDF_VISUAL_FILENAME,
    PNG_VISUAL_FILENAME,
    WORD_OUTPUT_FILENAME,
    ReviewerBundle,
)
from app.reporting.visual_diff import image_to_png_bytes

logger = get_logger(__name__)

_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")

# Filenames aligned with scripts/run_manual_compare.py
LLM_USAGE_FILENAME = "comparison_llm_usage.json"
ARTIFACT_FILENAMES = frozenset(
    {
        PDF_VISUAL_FILENAME,
        PNG_VISUAL_FILENAME,
        "visual_diff_report.html",
        "visual_diff_manifest.json",
        LLM_USAGE_FILENAME,
        WORD_OUTPUT_FILENAME,
    },
)

_MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# Remove job dirs older than this (seconds).
_COMPARE_JOB_TTL_SECONDS = 86400


def compare_jobs_root() -> Path:
    root = get_settings().data_dir / "tmp" / "compare_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_valid_job_id(job_id: str) -> bool:
    return bool(_JOB_ID_RE.match(job_id))


def cleanup_stale_compare_jobs(
    *,
    max_age_seconds: int = _COMPARE_JOB_TTL_SECONDS,
    now: float | None = None,
) -> None:
    """Delete job directories older than ``max_age_seconds`` (best-effort)."""
    root = compare_jobs_root()
    if not root.is_dir():
        return
    t = now if now is not None else time.time()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not is_valid_job_id(child.name):
            continue
        try:
            age = t - child.stat().st_mtime
        except OSError:
            continue
        if age <= max_age_seconds:
            continue
        try:
            shutil.rmtree(child, ignore_errors=True)
            logger.info("removed stale compare job dir=%s age_s=%.0f", child.name, age)
        except OSError:
            logger.warning("failed to remove stale compare job dir=%s", child.name)


def persist_reviewer_artifacts(
    job_id: str,
    bundle: ReviewerBundle,
    *,
    include_debug_artifacts: bool = False,
    llm_usage: dict[str, Any] | None = None,
) -> None:
    """Write reviewer-facing artifacts, with optional internal debug files."""
    if not is_valid_job_id(job_id):
        msg = f"invalid job_id: {job_id!r}"
        raise ValueError(msg)
    base = compare_jobs_root() / job_id
    base.mkdir(parents=True, exist_ok=False)
    (base / bundle.visual_filename).write_bytes(bundle.visual_bytes)
    (base / WORD_OUTPUT_FILENAME).write_bytes(bundle.comparison_docx)
    payload = llm_usage if isinstance(llm_usage, dict) else {}
    (base / LLM_USAGE_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if include_debug_artifacts:
        if bundle.visual_kind != "png":
            (base / PNG_VISUAL_FILENAME).write_bytes(image_to_png_bytes(bundle.overlay_bgr))
        (base / "visual_diff_manifest.json").write_text(
            bundle.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if bundle.html is not None:
            (base / "visual_diff_report.html").write_text(bundle.html, encoding="utf-8")
    cleanup_stale_compare_jobs()


def artifact_file_path(job_id: str, filename: str) -> Path | None:
    """Return absolute path if ``filename`` is allowed and exists."""
    if not is_valid_job_id(job_id):
        return None
    if filename not in ARTIFACT_FILENAMES:
        return None
    root = compare_jobs_root().resolve()
    path = (root / job_id / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path


def media_type_for_filename(filename: str) -> str:
    suf = Path(filename).suffix.lower()
    return _MEDIA_TYPES.get(suf, "application/octet-stream")
