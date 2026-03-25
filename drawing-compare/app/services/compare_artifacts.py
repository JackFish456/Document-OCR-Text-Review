"""Internal compare pipeline artifacts for downstream renderers and tooling."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.extraction import ExtractedField
from app.models.match import MatchResult
from app.preprocessing.types import PreprocessPageResult


@dataclass(slots=True)
class CompareArtifacts:
    """Non-API compare outputs retained for diagnostics and visual rendering."""

    source_pages: list[PreprocessPageResult]
    target_pages: list[PreprocessPageResult]
    results: list[MatchResult]
    source_fields: list[ExtractedField]
    target_fields: list[ExtractedField]

    @property
    def source_page(self) -> PreprocessPageResult:
        """First source page (backward compatible with single-page runs)."""
        if not self.source_pages:
            msg = "CompareArtifacts has no source pages"
            raise ValueError(msg)
        return self.source_pages[0]

    @property
    def target_page(self) -> PreprocessPageResult:
        """First target page (backward compatible with single-page runs)."""
        if not self.target_pages:
            msg = "CompareArtifacts has no target pages"
            raise ValueError(msg)
        return self.target_pages[0]

    @property
    def total_source(self) -> int:
        return len(self.source_fields)
