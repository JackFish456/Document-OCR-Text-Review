"""Word (.docx) export for drawing comparison: document title + findings table only."""

from __future__ import annotations

import io

from docx import Document
from docx.shared import Inches, Pt

from app.models.comparison import CompareResponse
from app.reporting.display import match_type_plain
from app.reporting.visual_diff import VisualDiffManifest

_DOC_TITLE = "Comparison report"


def build_comparison_docx_bytes(
    response: CompareResponse,
    manifest: VisualDiffManifest,
) -> bytes:
    """Build a .docx with a title and the findings table (no overview, review flags, or notes)."""
    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(11)
    for section in doc.sections:
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)

    title = str(response.extras.get("comparison_docx_title") or _DOC_TITLE).strip() or _DOC_TITLE
    doc.add_heading(title, level=0)

    annotations = sorted(manifest.annotations, key=lambda a: a.index)
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    headers = ("#", "Match type", "Source", "Target", "Confidence")
    for i, h in enumerate(headers):
        hdr[i].text = h
    for ann in annotations:
        row = table.add_row().cells
        row[0].text = str(ann.index)
        row[1].text = match_type_plain(ann.match_type)
        row[2].text = ann.source_text or "—"
        row[3].text = ann.target_text or "—"
        row[4].text = f"{ann.confidence:.2f}"

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
