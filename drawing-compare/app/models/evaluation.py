"""Schemas for golden dataset, manifests, and evaluation metrics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.golden import GoldenExpectedLabel


class GoldenPairManifestEntry(BaseModel):
    """One row in a golden pairs manifest."""

    pair_id: str = Field(min_length=1)
    drawing_a_ref: str = Field(
        min_length=1,
        description="Path to drawing A (source / baseline), usually relative to repo root",
    )
    drawing_b_ref: str = Field(
        min_length=1,
        description="Path to drawing B (target / revised), usually relative to repo root",
    )
    annotation_path: str | None = Field(
        default=None,
        description="Optional path to per-pair JSON under annotations/",
    )
    notes: str | None = Field(
        default=None,
        description="Why this pair belongs together; optional but recommended for new pairs",
    )


class GoldenManifestDocument(BaseModel):
    """Versioned manifest file (preferred on-disk format for new datasets)."""

    schema_version: int = Field(default=1, ge=1)
    manifest_id: str = Field(
        default="default",
        min_length=1,
        description="Stable id for this manifest (used in index and reports)",
    )
    description: str | None = None
    pairs: list[GoldenPairManifestEntry] = Field(default_factory=list)


class GoldenManifestIndexEntry(BaseModel):
    """Pointer to a manifest file from manifests/index.json."""

    file: str = Field(min_length=1, description="Filename under manifests/, e.g. train.json")
    description: str | None = None


class GoldenManifestIndex(BaseModel):
    """Registry of manifest files so new splits can be added without code changes."""

    schema_version: int = Field(default=1, ge=1)
    default_manifest: str | None = Field(
        default=None,
        description="Optional default manifest filename for tooling",
    )
    manifests: list[GoldenManifestIndexEntry] = Field(default_factory=list)


_LEGACY_FIELD_EXPECTED: dict[str, str] = {
    "exact_match": GoldenExpectedLabel.MATCHED.value,
    "partial_match": GoldenExpectedLabel.MATCHED.value,
    "changed_value": GoldenExpectedLabel.CHANGED.value,
    "missing_in_target": GoldenExpectedLabel.MISSING.value,
    "extra_in_target": GoldenExpectedLabel.EXTRA.value,
}


class GoldenFieldExpectation(BaseModel):
    """Expected golden label for one logical field in a pair."""

    field_id: str = Field(min_length=1)
    expected: GoldenExpectedLabel
    optional_notes: str | None = None

    @field_validator("expected", mode="before")
    @classmethod
    def _coerce_legacy_engine_labels(cls, v: object) -> object:
        if isinstance(v, str) and v in _LEGACY_FIELD_EXPECTED:
            return _LEGACY_FIELD_EXPECTED[v]
        return v


class GoldenPairAnnotation(BaseModel):
    """Per-pair ground truth loaded from annotations/."""

    schema_version: int = Field(default=1, ge=1)
    pair_id: str = Field(min_length=1)
    fields: list[GoldenFieldExpectation] = Field(default_factory=list)
    review_note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    pair_level_expected: GoldenExpectedLabel | None = Field(
        default=None,
        description="Optional coarse label for the whole pair",
    )


class FieldLevelMetric(BaseModel):
    """Micro-F1 style counts for one golden label."""

    label: GoldenExpectedLabel
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


class EvaluationRunSummary(BaseModel):
    """Aggregate metrics for an eval run."""

    run_id: str
    pair_count: int
    field_accuracy: float | None = None
    metrics_by_label: dict[str, FieldLevelMetric] = Field(default_factory=dict)
    macro_f1: float | None = None


class PerClassMetricDetail(BaseModel):
    """Precision / recall / F1 for one golden label (one-vs-rest style counts)."""

    label: str
    support: int = Field(ge=0, description="Gold instances with this label (tp + fn)")
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0.0)
    recall: float = Field(ge=0.0)
    f1: float = Field(ge=0.0)


class ConfusionMatrixReport(BaseModel):
    """Gold label (rows) vs predicted coarse bucket (columns)."""

    row_labels: list[str]
    column_labels: list[str]
    matrix: list[list[int]]


class BinaryTaskMetric(BaseModel):
    """Accuracy on a subset of fields (e.g. only those labeled *changed* in gold)."""

    task: str
    correct: int = Field(ge=0)
    total: int = Field(ge=0)
    accuracy: float | None = Field(
        default=None,
        description="correct / total when total > 0, else null",
    )


class PairEvaluationRecord(BaseModel):
    """One drawing pair in a batch eval run."""

    pair_id: str
    success: bool
    error_message: str | None = None
    gold_field_count: int = 0
    fields_correct: int | None = None
    fields_total: int | None = None


class BatchEvaluationReport(BaseModel):
    """Full batch evaluation output (JSON-serializable)."""

    run_id: str
    manifest_name: str
    project_root: str
    golden_root: str
    ocr_provider: str | None = None
    pairs_attempted: int
    pairs_succeeded: int
    pairs_failed: int
    total_gold_fields: int
    overall_field_accuracy: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    per_class: list[PerClassMetricDetail] = Field(default_factory=list)
    confusion_matrix: ConfusionMatrixReport
    changed_value_detection: BinaryTaskMetric
    missing_detection: BinaryTaskMetric
    pair_results: list[PairEvaluationRecord] = Field(default_factory=list)
    evaluation_run_summary: EvaluationRunSummary
