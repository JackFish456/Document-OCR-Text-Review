"""Validate and persist uploaded drawing files for comparison."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from fastapi import HTTPException, UploadFile

ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
)
SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_MAX_BYTES: Final[int] = 50 * 1024 * 1024


def safe_stored_filename(upload_name: str | None, fallback_stem: str) -> str:
    """Avoid path traversal; require a known extension."""
    raw = (upload_name or "").strip()
    base = Path(raw).name if raw else fallback_stem
    if not base or base in {".", ".."}:
        base = fallback_stem
    stem = SAFE_NAME_RE.sub("_", Path(base).stem)[:120] or fallback_stem
    suf = Path(base).suffix.lower()
    if suf not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported or missing file extension on {upload_name!r}. "
                f"Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )
    return f"{stem}{suf}"


def extension_ok(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


async def save_upload(
    upload: UploadFile,
    *,
    dest_dir: Path,
    fallback_stem: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Write upload to ``dest_dir``; raises HTTPException on validation failure."""
    name = safe_stored_filename(upload.filename, fallback_stem)
    dest = dest_dir / name
    if not extension_ok(dest):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stored path extension for {upload.filename!r}",
        )
    data = await upload.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"Empty file: {upload.filename!r}",
        )
    if len(data) > max_bytes:
        mb = max_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({upload.filename!r}); max {mb} MiB",
        )
    dest.write_bytes(data)
    return dest
