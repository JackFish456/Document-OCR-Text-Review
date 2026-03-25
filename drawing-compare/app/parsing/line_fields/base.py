"""Pluggable extractor protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.parsing.line_fields.context import LineExtractionContext
from app.parsing.line_fields.matches import ExtractionMatch


class LineFieldExtractor(ABC):
    """Regex- or heuristic-based extractor; may return zero, one, or many matches per line."""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Stable id for logging, tests, and ``ExtractedField.extraction_rule``."""

    @abstractmethod
    def extract(self, ctx: LineExtractionContext) -> list[ExtractionMatch]:
        """Return all matches for this line (possibly empty)."""
