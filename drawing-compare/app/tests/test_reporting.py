"""Reporting: JSON document, Markdown, and narrative."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.models.comparison import CompareResponse
from app.models.extraction import ExtractedField
from app.models.match import ComparisonReport, MatchResult, MatchType
from app.models.ocr import BoundingBox
from app.models.review_flag import ReviewFlag, ReviewFlagSeverity, ReviewFlagType
from app.reporting.json_report import build_comparison_json_document, comparison_report_json_str
from app.reporting.markdown_report import build_comparison_markdown
from app.reporting.narrative import build_executive_summary


def _box() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=1, y2=1)


def _sample_response() -> CompareResponse:
    src_m = ExtractedField(
        field_id="s1",
        field_type="revision",
        label="REV",
        value="A",
        normalized_label="rev",
        normalized_value="a",
        bbox=_box(),
        confidence=1.0,
    )
    tgt_m = ExtractedField(
        field_id="t1",
        field_type="revision",
        label="REV",
        value="B",
        normalized_label="rev",
        normalized_value="b",
        bbox=_box(),
        confidence=1.0,
    )
    only_src = ExtractedField(
        field_id="s2",
        field_type="generic",
        label="NOTE",
        value="Remove",
        normalized_label="note",
        normalized_value="remove",
        bbox=_box(),
        confidence=1.0,
    )
    only_tgt = ExtractedField(
        field_id="t2",
        field_type="generic",
        label="NOTE",
        value="Add",
        normalized_label="note",
        normalized_value="add",
        bbox=_box(),
        confidence=1.0,
    )
    results = [
        MatchResult(
            source_field=src_m,
            target_field=tgt_m,
            match_type=MatchType.CHANGED_VALUE,
            confidence=0.88,
            reason="classification=changed_value",
        ),
        MatchResult(
            source_field=only_src,
            target_field=None,
            match_type=MatchType.MISSING_IN_TARGET,
            confidence=1.0,
            reason="missing",
        ),
        MatchResult(
            source_field=None,
            target_field=only_tgt,
            match_type=MatchType.EXTRA_IN_TARGET,
            confidence=1.0,
            reason="extra",
        ),
    ]
    report = ComparisonReport.from_flat_results(results, total_source=2)
    flags = [
        ReviewFlag(
            type=ReviewFlagType.CHANGED_VALUE,
            severity=ReviewFlagSeverity.MEDIUM,
            field_reference="s1",
            reason="[rule:changed_value] demo",
            confidence=0.88,
        )
    ]
    return CompareResponse(
        comparison_id=uuid4(),
        report=report,
        review_flags=flags,
        extras={"mode": "test"},
    )


def test_executive_summary_plain_language() -> None:
    r = _sample_response()
    text = build_executive_summary(r.report, r.review_flags)
    assert "2 text field" in text or "2 text fields" in text
    assert "review item" in text.lower()


def test_json_document_sections() -> None:
    r = _sample_response()
    doc = build_comparison_json_document(r)
    assert doc["schema_version"] == "1.0"
    assert "executive_summary" in doc
    assert doc["summary_counts"]["missing_on_target_drawing"] == 1
    assert len(doc["changed_values"]) == 1
    assert len(doc["missing_fields"]) == 1
    assert len(doc["extra_fields"]) == 1
    assert len(doc["review_flags"]) == 1
    assert doc["review_flags"][0]["severity_explained"]
    assert "structured_comparison" in doc


def test_json_round_trip_string() -> None:
    r = _sample_response()
    s = comparison_report_json_str(r)
    assert '"executive_summary"' in s
    assert "comparison_id" in s


def test_markdown_sections_and_summary() -> None:
    r = _sample_response()
    md = build_comparison_markdown(r)
    assert "# Drawing comparison report" in md
    assert "## At a glance" in md
    assert "## Summary counts" in md
    assert "## Text that changed between drawings" in md
    assert "## Missing on the target drawing" in md
    assert "## Extra text on the target drawing" in md
    assert "## Review flags" in md
    assert "High" in md or "Medium" in md or "Low" in md
    assert "Match confidence" in md
    assert "OCR confidence" in md


def test_markdown_uncertain_section_present() -> None:
    r = _sample_response()
    assert "## Unclear matches (please review)" in build_comparison_markdown(r)


def test_markdown_includes_annotated_pdf_link_and_numbered_findings() -> None:
    r = _sample_response()
    findings = [
        SimpleNamespace(
            index=1,
            match_type=MatchType.CHANGED_VALUE,
            source_text="REV: A",
            target_text="REV: B",
            confidence=0.88,
            source_ocr_confidence=0.91,
            target_ocr_confidence=0.86,
        ),
        SimpleNamespace(
            index=2,
            match_type=MatchType.MISSING_IN_TARGET,
            source_text="NOTE: Remove",
            target_text="",
            confidence=1.0,
            source_ocr_confidence=0.93,
            target_ocr_confidence=None,
        ),
        SimpleNamespace(
            index=3,
            match_type=MatchType.EXTRA_IN_TARGET,
            source_text="",
            target_text="NOTE: Add",
            confidence=1.0,
            source_ocr_confidence=None,
            target_ocr_confidence=0.89,
        ),
    ]

    md = build_comparison_markdown(
        r,
        annotated_findings=findings,
        annotated_pdf_href="visual_diff_overlay.pdf",
    )

    assert "**Open annotated PDF:** [visual_diff_overlay.pdf](visual_diff_overlay.pdf)" in md
    assert "## Annotated findings" in md
    assert (
        "| Annotation # | Finding type | Source text | Target text | Match confidence | OCR confidence | Annotated visual |"
        in md
    )
    assert md.count("[Open PDF](visual_diff_overlay.pdf)") == 3
    assert md.index("| 1 | Changed text |") < md.index("| 2 | Missing on target |")
    assert md.index("| 2 | Missing on target |") < md.index("| 3 | Only on target |")
