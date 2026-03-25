"""Human review flags derived from comparison and OCR quality."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ReviewFlagType(StrEnum):
    MISSING_INFORMATION = "missing_information"
    CHANGED_VALUE = "changed_value"
    POSSIBLE_ERROR = "possible_error"
    AMBIGUOUS_MATCH = "ambiguous_match"
    EXTRA_INFORMATION = "extra_information"


class ReviewFlagSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewFlag(BaseModel):
    """Actionable item for manual QA workflows."""

    type: ReviewFlagType
    severity: ReviewFlagSeverity = ReviewFlagSeverity.MEDIUM
    field_reference: str | None = Field(
        default=None,
        description="Stable field id, token id, or pair key when applicable",
    )
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, description="Model-estimated flag confidence")

    def summary_line(self) -> str:
        return f"[{self.severity.value}] {self.type.value}: {self.reason}"
