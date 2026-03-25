"""Word (.docx) export for drawing comparison: overview, table, optional LLM narrative."""

from __future__ import annotations

import io
from typing import Any

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Pt

from app.models.comparison import CompareResponse
from app.models.match import ComparisonSummary
from app.reporting.display import flag_type_plain, match_type_plain, severity_plain
from app.reporting.visual_diff import VisualDiffManifest


def _deterministic_overview(summary: ComparisonSummary) -> str:
    """Short narrative when no LLM summary is available."""
    parts = [
        f"Source fields considered: {summary.total_source}.",
        f"Strong matches (exact or partial): {summary.matched}.",
        f"Changed values: {summary.changed}.",
        f"Missing on target: {summary.missing}.",
        f"Only on target: {summary.extra_target}.",
        f"Unclear matches: {summary.uncertain}.",
    ]
    return " ".join(parts)


def build_comparison_docx_bytes(
    response: CompareResponse,
    manifest: VisualDiffManifest,
) -> bytes:
    """Build a .docx with counts overview, optional LLM narrative, and annotation table."""
    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(11)

    title = doc.add_heading("Drawing comparison summary", level=0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    doc.add_paragraph(f"Comparison ID: {response.comparison_id}")

    extras: dict[str, Any] = response.extras
    llm_key = "comparison_llm_narrative_summary"
    narrative = extras.get(llm_key) if isinstance(extras.get(llm_key), str) else None
    summary = response.report.summary

    doc.add_heading("Overview", level=1)
    doc.add_paragraph(_deterministic_overview(summary))

    if narrative and narrative.strip():
        doc.add_heading("Executive summary", level=1)
        doc.add_paragraph(narrative.strip())

    if response.review_flags:
        doc.add_heading("Review flags", level=1)
        for flag in response.review_flags:
            line = (
                f"[{flag.severity.value}] {flag_type_plain(flag.type)} — "
                f"{flag.reason} ({severity_plain(flag.severity)})"
            )
            if flag.field_reference:
                line += f" (ref: {flag.field_reference})"
            doc.add_paragraph(line, style="List Bullet")

    annotations = sorted(manifest.annotations, key=lambda a: a.index)
    doc.add_heading("Detailed findings (matches PDF index numbers)", level=1)
    if not annotations:
        doc.add_paragraph("No visible differences were flagged on the overlay.")
    else:
        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        headers = ("#", "Match type", "Source", "Target", "Confidence", "Notes")
        for i, h in enumerate(headers):
            hdr[i].text = h
        for ann in annotations:
            row = table.add_row().cells
            row[0].text = str(ann.index)
            row[1].text = match_type_plain(ann.match_type)
            row[2].text = ann.source_text or "—"
            row[3].text = ann.target_text or "—"
            row[4].text = f"{ann.confidence:.2f}"
            notes = (ann.reason or "").strip()
            row[5].text = notes if notes else "—"

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
