"""Intermediate extraction results before building ``ExtractedField``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractionMatch:
    """Single semantic hit from one extractor on one line."""

    field_type: str
    label: str
    value: str
    raw_text: str
    rule_id: str
    rule_weight: float

    def combined_confidence(self, line_confidence: float) -> float:
        """Blend OCR line confidence with rule prior (both 0..1)."""
        lc = line_confidence if line_confidence > 0 else 0.85
        base = min(1.0, max(0.0, lc)) * min(1.0, max(0.0, self.rule_weight))
        return round(min(1.0, max(0.0, base)), 4)
