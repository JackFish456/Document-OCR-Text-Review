"""Batch evaluation framework: metrics merge, reporting, and runner wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.reporting import compose_batch_report, write_batch_eval_csvs, write_batch_eval_json
from app.evaluation.runner import run_batch_golden_evaluation
from app.models.comparison import CompareResponse
from app.models.extraction import ExtractedField
from app.models.golden import GoldenExpectedLabel
from app.models.match import ComparisonReport, MatchResult, MatchType
from app.models.ocr import BoundingBox


def test_merge_and_confusion_and_task_accuracy() -> None:
    a = EvaluationMetrics("a")
    a.update(GoldenExpectedLabel.MATCHED, MatchType.EXACT_MATCH)
    a.update(GoldenExpectedLabel.CHANGED, MatchType.CHANGED_VALUE)
    a.update(GoldenExpectedLabel.CHANGED, MatchType.EXACT_MATCH)
    a.update(GoldenExpectedLabel.MISSING, MatchType.MISSING_IN_TARGET)
    a.update(GoldenExpectedLabel.MISSING, MatchType.EXTRA_IN_TARGET)

    assert a.changed_value_detection_accuracy() == pytest.approx(0.5)
    assert a.missing_detection_accuracy() == pytest.approx(0.5)
    cm = a.confusion_matrix_report()
    assert "matched" in cm.row_labels and "none" in cm.column_labels
    assert sum(sum(row) for row in cm.matrix) == 5

    b = EvaluationMetrics("b")
    b.update(GoldenExpectedLabel.EXTRA, MatchType.EXTRA_IN_TARGET)
    a.merge_from(b)
    assert a.total_gold_fields() == 6
    assert a.summary().metrics_by_label["extra"].true_positives == 1


def test_compose_and_write_outputs(tmp_path: Path) -> None:
    m = EvaluationMetrics("agg")
    m.record_pair()
    m.update(GoldenExpectedLabel.MATCHED, MatchType.PARTIAL_MATCH)
    report = compose_batch_report(
        m,
        run_id="run1",
        manifest_name="m.json",
        project_root=tmp_path,
        golden_root=tmp_path / "golden",
        ocr_provider="stub",
        pair_results=[],
        pairs_attempted=1,
        pairs_succeeded=1,
        pairs_failed=0,
    )
    assert report.overall_field_accuracy == pytest.approx(1.0)
    assert report.changed_value_detection.total == 0
    assert report.missing_detection.total == 0

    out = tmp_path / "out"
    write_batch_eval_json(report, out / "eval_report.json")
    write_batch_eval_csvs(report, out)
    assert (out / "eval_report.json").is_file()
    assert (out / "eval_summary.csv").is_file()
    assert (out / "eval_per_class.csv").is_file()
    assert (out / "eval_confusion.csv").is_file()


def test_runner_uses_mock_compare_service() -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=1, y2=1)
    src_t = ExtractedField(
        field_id="title_project",
        value="P1",
        bbox=bbox,
        confidence=1.0,
    )
    tgt_t = ExtractedField(field_id="t1", value="P1", bbox=bbox, confidence=1.0)
    src_r = ExtractedField(field_id="revision", value="A", bbox=bbox, confidence=1.0)
    tgt_r = ExtractedField(field_id="t2", value="B", bbox=bbox, confidence=1.0)
    flat = [
        MatchResult(
            source_field=src_t,
            target_field=tgt_t,
            match_type=MatchType.EXACT_MATCH,
            confidence=1.0,
        ),
        MatchResult(
            source_field=src_r,
            target_field=tgt_r,
            match_type=MatchType.CHANGED_VALUE,
            confidence=1.0,
        ),
    ]
    report_model = ComparisonReport.from_flat_results(flat, total_source=2)
    response = CompareResponse(report=report_model, review_flags=[], extras={})

    mock_svc = MagicMock()
    mock_svc.compare_paths.return_value = response

    root = Path(__file__).resolve().parent.parent.parent
    golden = root / "data" / "golden"
    out = run_batch_golden_evaluation(
        "example_manifest.json",
        project_root=root,
        golden_root=golden,
        run_id="mock-run",
        service=mock_svc,
    )
    assert out.pairs_succeeded >= 1
    assert out.total_gold_fields >= 2
    assert out.overall_field_accuracy == pytest.approx(1.0)
    mock_svc.compare_paths.assert_called()
