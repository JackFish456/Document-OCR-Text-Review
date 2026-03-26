"""Word export for comparison reports."""

from __future__ import annotations

import io
import zipfile
from uuid import uuid4

from app.models.comparison import CompareResponse
from app.models.match import ComparisonReport, ComparisonSummary, MatchType
from app.models.ocr import BoundingBox
from app.reporting.comparison_docx import build_comparison_docx_bytes
from app.reporting.visual_diff import NormalizedBoundingBox, VisualAnnotation, VisualDiffManifest


def _box() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=10, y2=10)


def test_build_comparison_docx_zip_and_table_row() -> None:
    """.docx is a ZIP (PK); table includes annotation aligned with PDF index."""
    summary = ComparisonSummary(
        total_source=1,
        matched=0,
        changed=1,
        missing=0,
        extra_target=0,
        uncertain=0,
    )
    report = ComparisonReport(
        summary=summary,
        matches=[],
        missing=[],
        extra_target=[],
        uncertain=[],
    )
    response = CompareResponse(comparison_id=uuid4(), report=report, review_flags=[], extras={})
    ann = VisualAnnotation(
        annotation_id="ann-001",
        index=1,
        match_type=MatchType.CHANGED_VALUE,
        source_text="OLD",
        target_text="NEW",
        confidence=0.9,
        reason="test",
        overlay_bbox_normalized=NormalizedBoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2),
    )
    manifest = VisualDiffManifest(
        comparison_id=str(response.comparison_id),
        source_image_width=100,
        source_image_height=100,
        target_image_width=100,
        target_image_height=100,
        summary=summary,
        annotations=[ann],
    )
    raw = build_comparison_docx_bytes(response, manifest)
    assert raw[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        doc_xml = zf.read("word/document.xml").decode("utf-8")
    assert "OLD" in doc_xml and "NEW" in doc_xml
    assert "Comparison ID:" not in doc_xml
    assert "Review flags" not in doc_xml
    assert "<w:t>Notes</w:t>" not in doc_xml
    assert "Executive summary" not in doc_xml
    assert "Overview" not in doc_xml
    assert doc_xml.index("Drawing comparison summary") < doc_xml.index(
        "Detailed findings (matches PDF index numbers)"
    )
    assert "<w:t>Confidence</w:t>" in doc_xml
    assert "<w:t>Match type</w:t>" in doc_xml


def test_build_comparison_docx_no_annotations() -> None:
    summary = ComparisonSummary(
        total_source=0,
        matched=0,
        changed=0,
        missing=0,
        extra_target=0,
        uncertain=0,
    )
    report = ComparisonReport(
        summary=summary,
        matches=[],
        missing=[],
        extra_target=[],
        uncertain=[],
    )
    response = CompareResponse(report=report, review_flags=[], extras={})
    manifest = VisualDiffManifest(
        comparison_id="c1",
        source_image_width=8,
        source_image_height=8,
        target_image_width=8,
        target_image_height=8,
        summary=summary,
        annotations=[],
    )
    raw = build_comparison_docx_bytes(response, manifest)
    assert raw[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        doc_xml = zf.read("word/document.xml").decode("utf-8")
    assert "No visible differences" in doc_xml
