"""Metrics and runners for golden-set evaluation."""

from app.evaluation.metrics import (
    CONFUSION_PREDICTED_COLUMNS,
    EvaluationMetrics,
    predicted_coarse_bucket,
    score_field_predictions,
)
from app.evaluation.reporting import (
    compose_batch_report,
    per_class_details_from_summary,
    print_batch_eval_console,
    write_batch_eval_csvs,
    write_batch_eval_json,
)
from app.evaluation.error_analysis_runner import run_batch_error_analysis
from app.evaluation.runner import run_batch_golden_evaluation

__all__ = [
    "CONFUSION_PREDICTED_COLUMNS",
    "EvaluationMetrics",
    "compose_batch_report",
    "per_class_details_from_summary",
    "predicted_coarse_bucket",
    "print_batch_eval_console",
    "run_batch_error_analysis",
    "run_batch_golden_evaluation",
    "score_field_predictions",
    "write_batch_eval_csvs",
    "write_batch_eval_json",
]
