"""Field-level classification metrics vs golden labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from app.models.evaluation import (
    BinaryTaskMetric,
    ConfusionMatrixReport,
    EvaluationRunSummary,
    FieldLevelMetric,
    GoldenPairAnnotation,
)
from app.models.golden import GoldenExpectedLabel
from app.models.match import MatchResult, MatchType

# Predicted side of the confusion matrix (coarse label + pipeline outcomes).
CONFUSION_PREDICTED_COLUMNS: tuple[str, ...] = (
    *(lbl.value for lbl in GoldenExpectedLabel),
    "uncertain",
    "none",
)


def prediction_for_field(predictions: list[MatchResult], field_id: str) -> MatchResult | None:
    """Resolve a prediction row by source field id (or target-only for extras)."""
    for p in predictions:
        if p.source_field is not None and p.source_field.field_id == field_id:
            return p
        if p.source_field is None and p.target_field is not None and p.target_field.field_id == field_id:
            return p
    return None


def match_type_to_golden_label(mt: MatchType) -> GoldenExpectedLabel | None:
    """Map engine match type to the coarse golden label space (if applicable)."""
    if mt in (MatchType.EXACT_MATCH, MatchType.PARTIAL_MATCH):
        return GoldenExpectedLabel.MATCHED
    if mt == MatchType.CHANGED_VALUE:
        return GoldenExpectedLabel.CHANGED
    if mt == MatchType.MISSING_IN_TARGET:
        return GoldenExpectedLabel.MISSING
    if mt == MatchType.EXTRA_IN_TARGET:
        return GoldenExpectedLabel.EXTRA
    return None


def prediction_matches_golden(predicted: MatchType, gold: GoldenExpectedLabel) -> bool:
    """True when engine output satisfies the human golden label."""
    if gold == GoldenExpectedLabel.MATCHED:
        return predicted in (MatchType.EXACT_MATCH, MatchType.PARTIAL_MATCH)
    if gold == GoldenExpectedLabel.CHANGED:
        return predicted == MatchType.CHANGED_VALUE
    if gold == GoldenExpectedLabel.MISSING:
        return predicted == MatchType.MISSING_IN_TARGET
    if gold == GoldenExpectedLabel.EXTRA:
        return predicted == MatchType.EXTRA_IN_TARGET
    return False


def predicted_coarse_bucket(predicted: MatchType | None) -> str:
    """Bucket for confusion matrix columns (expected label space + uncertain + none)."""
    if predicted is None:
        return "none"
    coarse = match_type_to_golden_label(predicted)
    if coarse is not None:
        return coarse.value
    return "uncertain"


class EvaluationMetrics:
    """Accumulate counts for golden-label field evaluation."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._tp: dict[GoldenExpectedLabel, int] = defaultdict(int)
        self._fp: dict[GoldenExpectedLabel, int] = defaultdict(int)
        self._fn: dict[GoldenExpectedLabel, int] = defaultdict(int)
        self._pair_count = 0
        self._confusion: dict[tuple[str, str], int] = defaultdict(int)
        self._changed_gold: int = 0
        self._changed_correct: int = 0
        self._missing_gold: int = 0
        self._missing_correct: int = 0

    def record_pair(self) -> None:
        self._pair_count += 1

    def update(self, expected: GoldenExpectedLabel, predicted: MatchType | None) -> None:
        bucket = predicted_coarse_bucket(predicted)
        self._confusion[(expected.value, bucket)] += 1
        if expected == GoldenExpectedLabel.CHANGED:
            self._changed_gold += 1
            if predicted == MatchType.CHANGED_VALUE:
                self._changed_correct += 1
        if expected == GoldenExpectedLabel.MISSING:
            self._missing_gold += 1
            if predicted == MatchType.MISSING_IN_TARGET:
                self._missing_correct += 1
        if predicted is not None and prediction_matches_golden(predicted, expected):
            self._tp[expected] += 1
            return
        self._fn[expected] += 1
        if predicted is not None:
            coarse = match_type_to_golden_label(predicted)
            if coarse is not None:
                self._fp[coarse] += 1

    def merge_from(self, other: EvaluationMetrics) -> None:
        """Sum counts from another accumulator (e.g. merge per-pair metrics into a batch)."""
        self._pair_count += other._pair_count
        for label in GoldenExpectedLabel:
            self._tp[label] += other._tp[label]
            self._fp[label] += other._fp[label]
            self._fn[label] += other._fn[label]
        for key, n in other._confusion.items():
            self._confusion[key] += n
        self._changed_gold += other._changed_gold
        self._changed_correct += other._changed_correct
        self._missing_gold += other._missing_gold
        self._missing_correct += other._missing_correct

    def confusion_items(self) -> Iterator[tuple[tuple[str, str], int]]:
        return iter(self._confusion.items())

    def total_gold_fields(self) -> int:
        return sum(self._tp[label] + self._fn[label] for label in GoldenExpectedLabel)

    def total_correct(self) -> int:
        return sum(self._tp[label] for label in GoldenExpectedLabel)

    def changed_value_detection_accuracy(self) -> float | None:
        if self._changed_gold == 0:
            return None
        return self._changed_correct / self._changed_gold

    def missing_detection_accuracy(self) -> float | None:
        if self._missing_gold == 0:
            return None
        return self._missing_correct / self._missing_gold

    def confusion_matrix_report(self) -> ConfusionMatrixReport:
        rows = [lbl.value for lbl in GoldenExpectedLabel]
        cols = list(CONFUSION_PREDICTED_COLUMNS)
        ri = {r: i for i, r in enumerate(rows)}
        ci = {c: j for j, c in enumerate(cols)}
        mat = [[0] * len(cols) for _ in rows]
        for (r, c), n in self._confusion.items():
            i, j = ri.get(r), ci.get(c)
            if i is not None and j is not None:
                mat[i][j] += n
        return ConfusionMatrixReport(row_labels=rows, column_labels=cols, matrix=mat)

    def binary_changed_value_metric(self) -> BinaryTaskMetric:
        t = self._changed_gold
        c = self._changed_correct
        return BinaryTaskMetric(
            task="changed_value_detection",
            correct=c,
            total=t,
            accuracy=(c / t) if t else None,
        )

    def binary_missing_metric(self) -> BinaryTaskMetric:
        t = self._missing_gold
        c = self._missing_correct
        return BinaryTaskMetric(
            task="missing_detection",
            correct=c,
            total=t,
            accuracy=(c / t) if t else None,
        )

    def summary(self) -> EvaluationRunSummary:
        by_label: dict[str, FieldLevelMetric] = {}
        f1_scores: list[float] = []
        for label in GoldenExpectedLabel:
            tp = self._tp[label]
            fp = self._fp[label]
            fn = self._fn[label]
            by_label[label.value] = FieldLevelMetric(
                label=label, true_positives=tp, false_positives=fp, false_negatives=fn
            )
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if tp + fn > 0:
                f1_scores.append(f1)
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
        n = self.total_gold_fields()
        acc = self.total_correct() / n if n else None
        return EvaluationRunSummary(
            run_id=self.run_id,
            pair_count=self._pair_count,
            field_accuracy=acc,
            metrics_by_label=by_label,
            macro_f1=macro_f1,
        )


def score_field_predictions(
    gold: GoldenPairAnnotation,
    predictions: list[MatchResult],
) -> EvaluationMetrics:
    """Score predictions for one pair; aligns on golden field_id."""
    metrics = EvaluationMetrics(run_id=gold.pair_id)
    metrics.record_pair()
    for exp in gold.fields:
        pred = prediction_for_field(predictions, exp.field_id)
        pred_type = pred.match_type if pred is not None else None
        metrics.update(exp.expected, pred_type)
    return metrics
