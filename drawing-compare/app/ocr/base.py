"""OCR provider protocol and shared types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from app.models.ocr import OcrRegion

OcrImageInput = NDArray[np.uint8]


@runtime_checkable
class OcrProvider(Protocol):
    """Pluggable OCR backend (Tesseract, cloud APIs, etc.)."""

    provider_name: str

    def run(self, image: OcrImageInput, *, language_hint: str | None = None) -> list[OcrRegion]:
        """Run OCR on a single-channel or BGR image array."""
        ...
