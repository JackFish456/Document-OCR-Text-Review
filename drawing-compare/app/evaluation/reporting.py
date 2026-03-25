"""Serialize batch evaluation results to JSON, CSV, and console."""

from __future__ import annotations

import csv
from pathlib import Path

from app.models.evaluation import (
    BatchEvaluationReport,
    EvaluationRunSummary,
    PairEvaluationRecord,
    PerClassMetricDetail,
)
from app.models.golden import GoldenExpectedLabel
from app.evaluation.metrics import EvaluationMetrics


def per_class_details_from_summary(summary: EvaluationRunSummary) -> list[PerClassMetricDetail]:
    rows: list[PerClassMetricDetail] = []
    for label in GoldenExpectedLabel:
        m = summary.metrics_by_label[label.value]
        tp, fp, fn = m.true_positives, m.false_positives, m.false_negatives
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            PerClassMetricDetail(
                label=label.value,
                support=support,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=prec,
                recall=rec,
                f1=f1,
            )
        )
    return rows


def macro_avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compose_batch_report(
    merged: EvaluationMetrics,
    *,
    run_id: str,
    manifest_name: str,
    project_root: Path,
    golden_root: Path,
    ocr_provider: str | None,
    pair_results: list[PairEvaluationRecord],
    pairs_attempted: int,
    pairs_succeeded: int,
    pairs_failed: int,
) -> BatchEvaluationReport:
    summary = merged.summary()
    per_class = per_class_details_from_summary(summary)
    with_support = [d for d in per_class if d.support > 0]
    macro_p = macro_avg([d.precision for d in with_support])
    macro_r = macro_avg([d.recall for d in with_support])
    macro_f = macro_avg([d.f1 for d in with_support])
    return BatchEvaluationReport(
        run_id=run_id,
        manifest_name=manifest_name,
        project_root=str(project_root.resolve()),
        golden_root=str(golden_root.resolve()),
        ocr_provider=ocr_provider,
        pairs_attempted=pairs_attempted,
        pairs_succeeded=pairs_succeeded,
        pairs_failed=pairs_failed,
        total_gold_fields=merged.total_gold_fields(),
        overall_field_accuracy=summary.field_accuracy,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
        per_class=per_class,
        confusion_matrix=merged.confusion_matrix_report(),
        changed_value_detection=merged.binary_changed_value_metric(),
        missing_detection=merged.binary_missing_metric(),
        pair_results=pair_results,
        evaluation_run_summary=summary,
    )


def write_batch_eval_json(report: BatchEvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


def write_batch_eval_csvs(report: BatchEvaluationReport, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_summary.csv"
    per_class_path = out_dir / "eval_per_class.csv"
    confusion_path = out_dir / "eval_confusion.csv"

    summary_rows = [
        ("section", "metric", "value"),
        ("run", "run_id", report.run_id),
        ("run", "manifest_name", report.manifest_name),
        ("run", "ocr_provider", report.ocr_provider or ""),
        ("run", "pairs_attempted", str(report.pairs_attempted)),
        ("run", "pairs_succeeded", str(report.pairs_succeeded)),
        ("run", "pairs_failed", str(report.pairs_failed)),
        ("run", "total_gold_fields", str(report.total_gold_fields)),
        ("aggregate", "overall_field_accuracy", _fmt_float(report.overall_field_accuracy)),
        ("aggregate", "macro_precision", _fmt_float(report.macro_precision)),
        ("aggregate", "macro_recall", _fmt_float(report.macro_recall)),
        ("aggregate", "macro_f1", _fmt_float(report.macro_f1)),
        (
            "tasks",
            "changed_value_detection_accuracy",
            _fmt_float(report.changed_value_detection.accuracy),
        ),
        (
            "tasks",
            "changed_value_detection_correct",
            str(report.changed_value_detection.correct),
        ),
        (
            "tasks",
            "changed_value_detection_total",
            str(report.changed_value_detection.total),
        ),
        ("tasks", "missing_detection_accuracy", _fmt_float(report.missing_detection.accuracy)),
        ("tasks", "missing_detection_correct", str(report.missing_detection.correct)),
        ("tasks", "missing_detection_total", str(report.missing_detection.total)),
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(summary_rows)

    with per_class_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "support",
                "precision",
                "recall",
                "f1",
                "true_positives",
                "false_positives",
                "false_negatives",
            ]
        )
        for d in report.per_class:
            w.writerow(
                [
                    d.label,
                    d.support,
                    f"{d.precision:.6f}",
                    f"{d.recall:.6f}",
                    f"{d.f1:.6f}",
                    d.true_positives,
                    d.false_positives,
                    d.false_negatives,
                ]
            )

    cm = report.confusion_matrix
    with confusion_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expected"] + cm.column_labels)
        for i, row_label in enumerate(cm.row_labels):
            w.writerow(
                [row_label] + [str(cm.matrix[i][j]) for j in range(len(cm.column_labels))],
            )

    return summary_path, per_class_path, confusion_path


def _fmt_float(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}"


def print_batch_eval_console(report: BatchEvaluationReport) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DRAWING COMPARE — GOLDEN EVALUATION")
    lines.append("=" * 72)
    lines.append(f"Run id:           {report.run_id}")
    lines.append(f"Manifest:         {report.manifest_name}")
    lines.append(f"OCR provider:     {report.ocr_provider or '(default)'}")
    lines.append(f"Pairs OK / total: {report.pairs_succeeded} / {report.pairs_attempted}  "
                 f"(failed: {report.pairs_failed})")
    lines.append(f"Gold fields:      {report.total_gold_fields}")
    lines.append("")
    lines.append("--- Aggregate ---")
    lines.append(f"Overall field accuracy: {_fmt_float(report.overall_field_accuracy)}")
    lines.append(f"Macro precision:        {_fmt_float(report.macro_precision)}")
    lines.append(f"Macro recall:           {_fmt_float(report.macro_recall)}")
    lines.append(f"Macro F1:               {_fmt_float(report.macro_f1)}")
    lines.append("")
    lines.append("--- Task-specific (subset accuracy) ---")
    ch = report.changed_value_detection
    lines.append(
        f"Changed-value detection: {_fmt_float(ch.accuracy)}  "
        f"({ch.correct}/{ch.total} gold-labeled *changed*)"
    )
    ms = report.missing_detection
    lines.append(
        f"Missing detection:       {_fmt_float(ms.accuracy)}  "
        f"({ms.correct}/{ms.total} gold-labeled *missing*)"
    )
    lines.append("")
    lines.append("--- Per-class (precision / recall / F1) ---")
    header = f"{'label':<10} {'sup':>5} {'prec':>8} {'recall':>8} {'f1':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for d in report.per_class:
        lines.append(
            f"{d.label:<10} {d.support:>5} {d.precision:>8.4f} {d.recall:>8.4f} {d.f1:>8.4f}"
        )
    lines.append("")
    lines.append("--- Confusion matrix (expected → predicted bucket) ---")
    cm = report.confusion_matrix
    colw = max(8, max(len(c) for c in cm.column_labels) + 1)

    def pad(s: str) -> str:
        return f"{s:<{colw}}"

    lines.append(pad("") + "".join(pad(c) for c in cm.column_labels))
    for i, rlab in enumerate(cm.row_labels):
        lines.append(pad(rlab) + "".join(f"{cm.matrix[i][j]:<{colw}}" for j in range(len(cm.column_labels))))
    lines.append("")
    if report.pairs_failed and any(not p.success for p in report.pair_results):
        lines.append("--- Pair errors ---")
        for p in report.pair_results:
            if not p.success and p.error_message:
                lines.append(f"  {p.pair_id}: {p.error_message}")
        lines.append("")
    lines.append("=" * 72)
    print("\n".join(lines))
