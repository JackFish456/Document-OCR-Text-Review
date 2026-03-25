"""Internal compare pipeline artifacts for downstream renderers and tooling."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.extraction import ExtractedField
from app.models.match import MatchResult
from app.preprocessing.types import PreprocessPageResult


@dataclass(slots=True)
class CompareArtifacts:
    """Non-API compare outputs retained for diagnostics and visual rendering."""

    source_page: PreprocessPageResult
    target_page: PreprocessPageResult
    results: list[MatchResult]
    source_fields: list[ExtractedField]
    target_fields: list[ExtractedField]

    @property
    def total_source(self) -> int:
        return len(self.source_fields)
