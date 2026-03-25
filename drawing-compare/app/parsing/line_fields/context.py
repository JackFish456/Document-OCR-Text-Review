"""Context passed to line field extractors."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.ocr import OCRLine


@dataclass(frozen=True, slots=True)
class LineExtractionContext:
    """One OCR line on a page, ready for rule-based parsing."""

    page_number: int
    line: OCRLine

    @property
    def text(self) -> str:
        return (self.line.text or "").strip()

    @property
    def line_confidence(self) -> float:
        """Mean token / line OCR confidence; falls back when providers omit it."""
        c = self.line.confidence
        if c and c > 0:
            return float(c)
        if self.line.tokens:
            confs = [t.confidence for t in self.line.tokens if t.confidence is not None]
            if confs:
                return sum(confs) / len(confs)
        return 0.85
