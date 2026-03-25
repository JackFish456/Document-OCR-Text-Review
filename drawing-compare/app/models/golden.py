"""Golden dataset schema: human-facing labels and pair records."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class GoldenExpectedLabel(StrEnum):
    """Expected outcome for a field or whole pair (evaluation ground truth).

    These four labels are stable for dataset growth; the comparison engine uses
    finer :class:`~app.models.match.MatchType` values which map onto these for scoring.
    """

    MATCHED = "matched"
    CHANGED = "changed"
    MISSING = "missing"
    EXTRA = "extra"


# Backward-compatible alias used in older docs and tests.
GoldenPairExpected = GoldenExpectedLabel


class GoldenPair(BaseModel):
    """Labeled source/target file pair for evaluation (simple CSV-style record)."""

    pair_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    target_file: str = Field(min_length=1)
    expected: GoldenExpectedLabel

    def manifests_paths_relative(self) -> tuple[str, str]:
        """Convenience for CSV/manifest exporters."""
        return self.source_file, self.target_file
