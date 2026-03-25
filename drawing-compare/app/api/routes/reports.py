"""Read-only routes for experiment artifacts (e.g. PDF text diff HTML)."""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.core.config import get_settings

router = APIRouter(prefix="/reports", tags=["reports"])

_NO_REPORT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>No text diff report</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      max-width: 52rem;
      margin: 2rem auto;
      padding: 0 1rem;
      color: #1a1a1a;
    }}
    code {{ background: #f4f4f5; padding: 0.15rem 0.35rem; border-radius: 4px; }}
    pre {{ background: #f4f4f5; padding: 1rem; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>No text diff report found</h1>
  <p>
    There is no <code>text_diff_report.html</code> under
    <code>experiments/manual_runs/textdiff_*</code> yet.
  </p>
  <p>Generate one from the repository root (Windows PowerShell):</p>
  <pre>{cmd}</pre>
  <p>Then reload this page. Reports are picked by newest file time under matching run folders.</p>
</body>
</html>
"""


def _repo_root() -> Path:
    return get_settings().data_dir.resolve().parent


def find_latest_text_diff_html(manual_runs: Path) -> Path | None:
    """Newest ``text_diff_report.html`` under ``textdiff_*`` run directories."""
    if not manual_runs.is_dir():
        return None
    found: list[tuple[float, Path]] = []
    for d in manual_runs.iterdir():
        if not d.is_dir() or not d.name.startswith("textdiff_"):
            continue
        report = d / "text_diff_report.html"
        if report.is_file():
            found.append((report.stat().st_mtime, report))
    if not found:
        return None
    found.sort(key=lambda x: -x[0])
    return found[0][1]


@router.get("/latest-text-diff", response_class=HTMLResponse)
def latest_text_diff_report() -> HTMLResponse:
    """Serve the latest PDF text diff HTML from manual runs, or instructions if missing."""
    manual_runs = _repo_root() / "experiments" / "manual_runs"
    latest = find_latest_text_diff_html(manual_runs)
    if latest is None:
        cmd = html.escape(
            "python scripts/compare_pdf_text.py ^\n"
            "  --source data/test_inputs/source.pdf ^\n"
            "  --target data/test_inputs/target.pdf",
            quote=False,
        )
        body = _NO_REPORT_HTML.format(cmd=cmd)
        return HTMLResponse(content=body, status_code=200)

    content = latest.read_text(encoding="utf-8")
    return HTMLResponse(content=content)
