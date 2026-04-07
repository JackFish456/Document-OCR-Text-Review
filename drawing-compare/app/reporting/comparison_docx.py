"""Word (.docx) export for drawing comparison: overview, table, optional LLM narrative."""

from __future__ import annotations

import io

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Inches, Pt

from app.models.comparison import CompareResponse
from app.reporting.display import format_confidence, format_ocr_confidence_pair, match_type_plain
from app.reporting.visual_diff import VisualDiffManifest


def _confidence_stub(confidence: float | None) -> object | None:
    if confidence is None:
        return None
    return type("_ConfidenceOnly", (), {"confidence": confidence})()


def build_comparison_docx_bytes(
    response: CompareResponse,
    manifest: VisualDiffManifest,
) -> bytes:
    """Build a .docx with findings table content for reviewers."""
    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(11)
    for section in doc.sections:
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)

    title = doc.add_heading("Drawing comparison summary", level=0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    annotations = sorted(manifest.annotations, key=lambda a: a.index)
    doc.add_heading("Detailed findings (matches PDF index numbers)", level=1)
    if not annotations:
        doc.add_paragraph("No visible differences were flagged on the overlay.")
    else:
        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        headers = ("#", "Match type", "Source", "Target", "Match confidence", "OCR confidence")
        for i, h in enumerate(headers):
            hdr[i].text = h
        for ann in annotations:
            row = table.add_row().cells
            row[0].text = str(ann.index)
            row[1].text = match_type_plain(ann.match_type)
            row[2].text = ann.source_text or "—"
            row[3].text = ann.target_text or "—"
            row[4].text = format_confidence(ann.confidence)
            row[5].text = format_ocr_confidence_pair(
                _confidence_stub(ann.source_ocr_confidence),
                _confidence_stub(ann.target_ocr_confidence),
            )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
