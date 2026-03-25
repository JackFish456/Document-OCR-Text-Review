"""Stub OCR provider returning no regions; use for API wiring tests."""

from __future__ import annotations

from app.models.ocr import BoundingBox, OcrRegion
from app.ocr.base import OcrImageInput


class StubOcrProvider:
    """Returns empty OCR output. Replace with real provider in composition root."""

    provider_name: str = "stub"

    def run(self, image: OcrImageInput, *, language_hint: str | None = None) -> list[OcrRegion]:
        _ = image, language_hint
        return []


class EchoOcrProvider:
    """Test helper: single synthetic region covering full image shape."""

    provider_name: str = "echo"

    def run(self, image: OcrImageInput, *, language_hint: str | None = None) -> list[OcrRegion]:
        _ = language_hint
        h, w = int(image.shape[0]), int(image.shape[1])
        return [
            OcrRegion(
                text="STUB",
                confidence=0.99,
                bbox=BoundingBox.from_xywh(0.0, 0.0, float(w), float(h)),
            )
        ]
