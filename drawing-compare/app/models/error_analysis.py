"""Schemas for golden-set error analysis (failure taxonomy and reports)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.models.golden import GoldenExpectedLabel


class ErrorCategory(StrEnum):
    """High-level failure bucket for prioritizing improvements."""

    OCR = "ocr"
    EXTRACTION = "extraction"
    MATCHING = "matching"
    CLASSIFICATION = "classification"


class FieldFailureRecord(BaseModel):
    """One gold field that did not match the engine output."""

    pair_id: str
    field_id: str
    category: ErrorCategory
    failure_pattern: str = Field(
        description="Stable key for grouping, e.g. gold_matched_pred_missing_in_target",
    )
    gold_expected: GoldenExpectedLabel
    predicted_match_type: str | None = Field(
        default=None,
        description="Engine MatchType.value, or null if no row",
    )
    predicted_bucket: str | None = Field(
        default=None,
        description="Coarse bucket: matched|changed|missing|extra|uncertain|none",
    )
    match_confidence: float | None = None
    engine_reason: str | None = None
    source_value_preview: str | None = None
    target_value_preview: str | None = None
    min_field_confidence: float | None = Field(
        default=None,
        description="Min of source/target ExtractedField.confidence when both exist",
    )
    in_source_extraction: bool = False
    in_target_extraction: bool = False
    rationale: str = Field(
        default="",
        description="One-line explanation for report readers",
    )


class PipelineFailureRecord(BaseModel):
    """Compare pipeline could not complete for a manifest row."""

    pair_id: str
    error_message: str


class ErrorAnalysisReport(BaseModel):
    """Aggregated error analysis for a batch run."""

    run_id: str
    manifest_name: str
    project_root: str
    golden_root: str
    ocr_provider: str | None = None
    ocr_confidence_threshold: float
    pairs_manifest_no_annotation: int = 0
    pairs_pipeline_failed: int = 0
    pipeline_failures: list[PipelineFailureRecord] = Field(default_factory=list)
    pairs_succeeded: int
    pairs_with_failures: int
    total_gold_fields: int
    total_failures: int
    failures_by_category: dict[str, int]
    pattern_counts: dict[str, int] = Field(
        default_factory=dict,
        description="failure_pattern -> count (entire batch)",
    )
    top_failure_patterns: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Sorted (pattern, count) descending",
    )
    examples_by_category: dict[str, list[FieldFailureRecord]] = Field(default_factory=dict)
    failures: list[FieldFailureRecord] = Field(
        default_factory=list,
        description="All failures (may be large; trim at generation time if needed)",
    )
