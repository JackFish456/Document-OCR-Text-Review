"""OCR text normalization for engineering drawings."""

from app.text_normalization.drawing import (
    create_canonical_key,
    normalize_label,
    normalize_line,
    normalize_value,
)
from app.text_normalization.types import NormalizedText

__all__ = [
    "NormalizedText",
    "create_canonical_key",
    "normalize_label",
    "normalize_line",
    "normalize_value",
]
