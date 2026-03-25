"""Types for OCR text normalization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Original OCR/string content plus drawing-normalized form."""

    raw: str
    normalized: str
