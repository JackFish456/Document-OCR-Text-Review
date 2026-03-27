"""HTTP API request/response wrappers for drawing comparison."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.match import ComparisonReport
from app.models.review_flag import ReviewFlag


class CompareRequest(BaseModel):
    """Input for a pairwise drawing comparison."""

    drawing_a_uri: str | None = Field(
        default=None,
        description="Reference to drawing A (HTTP(S), file path, or internal ID resolver).",
    )
    drawing_b_uri: str | None = Field(default=None, description="Reference to drawing B.")
    drawing_a_id: str | None = None
    drawing_b_id: str | None = None
    job_metadata: dict[str, Any] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    """API response: aggregate report plus review flags and diagnostics."""

    comparison_id: UUID = Field(default_factory=uuid4)
    report: ComparisonReport
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)


class BatchCompareRequest(BaseModel):
    """Compare one baseline drawing against many candidate drawings (server-local paths)."""

    baseline_uri: str = Field(
        ...,
        description="Path to baseline (original) drawing visible to the API process.",
    )
    candidate_uris: list[str] = Field(
        ...,
        min_length=1,
        description="Paths to candidate drawings (merged PDF order preserved).",
    )
    candidate_labels: list[str] | None = Field(
        default=None,
        description="Optional labels parallel to candidate_uris (same length when set).",
    )
    job_metadata: dict[str, Any] = Field(default_factory=dict)
    fail_fast: bool = Field(
        default=False,
        description="If true, the entire batch fails when any pairwise compare fails.",
    )
    include_text_reports: bool = Field(
        default=True,
        description=(
            "If true, each successful pair includes comparison_json_report and "
            "comparison_markdown_report in pair_extras."
        ),
    )
    ocr_provider: str | None = Field(
        default=None,
        description="Optional OCR provider override (same semantics as multipart /compare).",
    )
    preprocess_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON object of PreprocessConfig fields merged over server defaults.",
    )

    @field_validator("candidate_uris")
    @classmethod
    def _non_empty_candidate_strings(cls, v: list[str]) -> list[str]:
        out = [str(u).strip() for u in v]
        if not out or any(not u for u in out):
            raise ValueError("candidate_uris must contain only non-empty path strings")
        return out

    @field_validator("baseline_uri")
    @classmethod
    def _baseline_non_empty(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("baseline_uri must be non-empty")
        return s

    @model_validator(mode="after")
    def _labels_parallel_to_candidates(self) -> BatchCompareRequest:
        if self.candidate_labels is not None and len(self.candidate_labels) != len(
            self.candidate_uris,
        ):
            raise ValueError(
                "candidate_labels must be the same length as candidate_uris when provided",
            )
        return self


class BatchPairResult(BaseModel):
    """Result for one baseline↔candidate pairwise run within a batch."""

    candidate_uri: str
    candidate_label: str | None = None
    ok: bool
    error: str | None = None
    comparison_id: UUID | None = None
    report: ComparisonReport | None = None
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    pair_extras: dict[str, Any] = Field(default_factory=dict)


class BatchCompareResponse(BaseModel):
    """Batch API response: per-pair outcomes plus one stapled annotated PDF in extras."""

    batch_id: UUID = Field(default_factory=uuid4)
    baseline_uri: str
    pair_count: int = Field(ge=0)
    pairs_succeeded: int = Field(ge=0)
    pairs: list[BatchPairResult]
    extras: dict[str, Any] = Field(default_factory=dict)
