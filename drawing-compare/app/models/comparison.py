"""HTTP API request/response wrappers for drawing comparison."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

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
