"""Field-level match outcomes and comparison report aggregates."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator

from app.models.extraction import ExtractedField


class MatchType(StrEnum):
    EXACT_MATCH = "exact_match"
    CHANGED_VALUE = "changed_value"
    PARTIAL_MATCH = "partial_match"
    MISSING_IN_TARGET = "missing_in_target"
    EXTRA_IN_TARGET = "extra_in_target"
    UNCERTAIN = "uncertain"

    def requires_both_fields(self) -> bool:
        """True when a `MatchResult` must carry both source and target `ExtractedField`."""
        return self not in (MatchType.MISSING_IN_TARGET, MatchType.EXTRA_IN_TARGET)


class MatchResult(BaseModel):
    """Alignment outcome for one source↔target field pair (or singleton)."""

    source_field: ExtractedField | None = None
    target_field: ExtractedField | None = None
    match_type: MatchType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None

    @model_validator(mode="after")
    def _presence_matches_type(self) -> Self:
        if self.match_type == MatchType.MISSING_IN_TARGET:
            if self.source_field is None or self.target_field is not None:
                raise ValueError("missing_in_target requires source_field and no target_field")
        elif self.match_type == MatchType.EXTRA_IN_TARGET:
            if self.target_field is None or self.source_field is not None:
                raise ValueError("extra_in_target requires target_field and no source_field")
        else:
            if self.source_field is None or self.target_field is None:
                raise ValueError(f"{self.match_type} requires both source_field and target_field")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def match_key(self) -> str:
        sid = self.source_field.field_id if self.source_field else ""
        tid = self.target_field.field_id if self.target_field else ""
        return f"{sid}::{tid}"


class ComparisonSummary(BaseModel):
    """Roll-up counts for one comparison run."""

    total_source: int = Field(ge=0, description="Fields on source drawing considered for match")
    matched: int = Field(
        ge=0,
        description="Strong matches: exact_match + partial_match",
    )
    changed: int = Field(ge=0, description="changed_value")
    missing: int = Field(ge=0, description="missing_in_target")
    extra_target: int = Field(ge=0, description="extra_in_target")
    uncertain: int = Field(ge=0)

    @classmethod
    def from_match_results(
        cls,
        results: list[MatchResult],
        *,
        total_source: int,
    ) -> ComparisonSummary:
        matched = sum(
            1 for r in results if r.match_type in (MatchType.EXACT_MATCH, MatchType.PARTIAL_MATCH)
        )
        changed = sum(1 for r in results if r.match_type == MatchType.CHANGED_VALUE)
        missing = sum(1 for r in results if r.match_type == MatchType.MISSING_IN_TARGET)
        extra = sum(1 for r in results if r.match_type == MatchType.EXTRA_IN_TARGET)
        uncertain = sum(1 for r in results if r.match_type == MatchType.UNCERTAIN)
        return cls(
            total_source=total_source,
            matched=matched,
            changed=changed,
            missing=missing,
            extra_target=extra,
            uncertain=uncertain,
        )


class ComparisonReport(BaseModel):
    """Structured comparison suitable for API export and golden-set evaluation."""

    summary: ComparisonSummary
    matches: list[MatchResult] = Field(
        default_factory=list,
        description="Aligned pairs: exact, partial, or changed",
    )
    missing: list[MatchResult] = Field(default_factory=list)
    extra_target: list[MatchResult] = Field(default_factory=list)
    uncertain: list[MatchResult] = Field(default_factory=list)

    @classmethod
    def from_flat_results(
        cls,
        results: list[MatchResult],
        *,
        total_source: int,
    ) -> ComparisonReport:
        """Partition flat MatchResult list into report sections and build summary."""
        matches: list[MatchResult] = []
        missing: list[MatchResult] = []
        extra: list[MatchResult] = []
        uncertain: list[MatchResult] = []
        for r in results:
            if r.match_type in (
                MatchType.EXACT_MATCH,
                MatchType.PARTIAL_MATCH,
                MatchType.CHANGED_VALUE,
            ):
                matches.append(r)
            elif r.match_type == MatchType.MISSING_IN_TARGET:
                missing.append(r)
            elif r.match_type == MatchType.EXTRA_IN_TARGET:
                extra.append(r)
            elif r.match_type == MatchType.UNCERTAIN:
                uncertain.append(r)
        summary = ComparisonSummary.from_match_results(results, total_source=total_source)
        return cls(
            summary=summary,
            matches=matches,
            missing=missing,
            extra_target=extra,
            uncertain=uncertain,
        )

    def all_results_flat(self) -> list[MatchResult]:
        return [*self.matches, *self.missing, *self.extra_target, *self.uncertain]
