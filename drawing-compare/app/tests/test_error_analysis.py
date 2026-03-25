"""Error categorization and error-analysis reporting."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.error_analysis_reporting import build_error_analysis_markdown, write_error_analysis_json
from app.evaluation.error_analysis_runner import build_error_analysis_report, collect_failures_for_pair
from app.evaluation.error_categorize import categorize_field_failure
from app.models.diagnostics import PipelineDiagnostics
from app.models.error_analysis import ErrorCategory, PipelineFailureRecord
from app.models.evaluation import GoldenFieldExpectation, GoldenPairAnnotation
from app.models.extraction import ExtractedField
from app.models.golden import GoldenExpectedLabel
from app.models.match import MatchResult, MatchType
from app.models.ocr import BoundingBox


def _bbox() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=1, y2=1)


def test_categorize_extraction_no_row() -> None:
    diag = PipelineDiagnostics(source_fields=[], target_fields=[])
    r = categorize_field_failure(
        "p1",
        "f1",
        GoldenExpectedLabel.MATCHED,
        None,
        diag,
        ocr_confidence_threshold=0.55,
    )
    assert r is not None
    assert r.category == ErrorCategory.EXTRACTION
    assert r.failure_pattern == "gold_field_not_in_source_extraction"


def test_categorize_matching_uncertain() -> None:
    src = ExtractedField(field_id="f1", value="a", bbox=_bbox(), confidence=0.99)
    tgt = ExtractedField(field_id="t1", value="b", bbox=_bbox(), confidence=0.99)
    pred = MatchResult(
        source_field=src,
        target_field=tgt,
        match_type=MatchType.UNCERTAIN,
        confidence=0.5,
    )
    diag = PipelineDiagnostics(source_fields=[src], target_fields=[tgt])
    r = categorize_field_failure("p", "f1", GoldenExpectedLabel.MATCHED, pred, diag)
    assert r is not None
    assert r.category == ErrorCategory.MATCHING


def test_categorize_classification_matched_vs_changed() -> None:
    src = ExtractedField(field_id="f1", value="a", bbox=_bbox(), confidence=0.99)
    tgt = ExtractedField(field_id="t1", value="b", bbox=_bbox(), confidence=0.99)
    pred = MatchResult(
        source_field=src,
        target_field=tgt,
        match_type=MatchType.CHANGED_VALUE,
        confidence=0.95,
    )
    diag = PipelineDiagnostics(source_fields=[src], target_fields=[tgt])
    r = categorize_field_failure("p", "f1", GoldenExpectedLabel.MATCHED, pred, diag)
    assert r is not None
    assert r.category == ErrorCategory.CLASSIFICATION
    assert r.failure_pattern == "gold_matched_pred_changed_value"


def test_categorize_ocr_low_confidence() -> None:
    src = ExtractedField(field_id="f1", value="a", bbox=_bbox(), confidence=0.2)
    tgt = ExtractedField(field_id="t1", value="b", bbox=_bbox(), confidence=0.99)
    pred = MatchResult(
        source_field=src,
        target_field=tgt,
        match_type=MatchType.CHANGED_VALUE,
        confidence=0.95,
    )
    diag = PipelineDiagnostics(source_fields=[src], target_fields=[tgt])
    r = categorize_field_failure("p", "f1", GoldenExpectedLabel.MATCHED, pred, diag, ocr_confidence_threshold=0.55)
    assert r is not None
    assert r.category == ErrorCategory.OCR


def test_collect_and_build_report(tmp_path: Path) -> None:
    ann = GoldenPairAnnotation(
        pair_id="g1",
        fields=[GoldenFieldExpectation(field_id="f1", expected=GoldenExpectedLabel.MATCHED)],
    )
    src = ExtractedField(field_id="f1", value="x", bbox=_bbox(), confidence=0.99)
    tgt = ExtractedField(field_id="t1", value="y", bbox=_bbox(), confidence=0.99)
    preds = [
        MatchResult(
            source_field=src,
            target_field=tgt,
            match_type=MatchType.CHANGED_VALUE,
            confidence=1.0,
        )
    ]
    diag = PipelineDiagnostics(source_fields=[src], target_fields=[tgt])
    fails = collect_failures_for_pair("g1", ann, preds, diag, ocr_confidence_threshold=0.55)
    assert len(fails) == 1
    report = build_error_analysis_report(
        fails,
        run_id="r1",
        manifest_name="m.json",
        project_root=tmp_path,
        golden_root=tmp_path,
        ocr_provider=None,
        ocr_confidence_threshold=0.55,
        pairs_manifest_no_annotation=0,
        pairs_pipeline_failed=1,
        pipeline_failures=[PipelineFailureRecord(pair_id="x", error_message="e")],
        pairs_succeeded=1,
        total_gold_fields_scored=1,
    )
    assert report.total_failures == 1
    assert report.failures_by_category["classification"] == 1
    md = build_error_analysis_markdown(report)
    assert "classification" in md
    write_error_analysis_json(report, tmp_path / "ea.json")
    assert (tmp_path / "ea.json").is_file()
