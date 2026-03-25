"""Adapter: file-based OCR providers -> ndarray OCR provider protocol."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.models.ocr import OCRDocument, OcrRegion
from app.ocr.base import OcrImageInput, OcrProvider
from app.ocr.document_provider import OCRProvider as DocumentOCRProvider


class DocumentProviderAdapter(OcrProvider):
    """Bridge file-based providers into the tile/page ndarray OCR pipeline."""

    def __init__(self, provider: DocumentOCRProvider, *, scratch_dir: Path) -> None:
        self._provider = provider
        self._scratch_dir = scratch_dir

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    def run(self, image: OcrImageInput, *, language_hint: str | None = None) -> list[OcrRegion]:
        _ = language_hint
        arr = _as_uint8_bgr(image)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        img_path = self._scratch_dir / f"{self.provider_name}_{uuid4().hex}.png"
        ok, encoded = cv2.imencode(".png", arr)
        if not ok:
            raise ValueError("Failed to encode tile image for OCR provider adapter")
        img_path.write_bytes(encoded.tobytes())
        try:
            doc = self._provider.extract(str(img_path))
        finally:
            try:
                img_path.unlink(missing_ok=True)
            except OSError:
                # Windows/AV/OneDrive can transiently lock temp files; ignore cleanup miss.
                pass
        return _regions_from_document(doc)


def _as_uint8_bgr(image: OcrImageInput) -> OcrImageInput:
    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image shape for OCR adapter: {arr.shape!r}")


def _regions_from_document(doc: OCRDocument) -> list[OcrRegion]:
    out: list[OcrRegion] = []
    for page in doc.pages:
        if page.lines:
            for line in page.lines:
                text = (line.text or "").strip()
                if not text:
                    continue
                out.append(
                    OcrRegion(
                        text=text,
                        confidence=float(line.confidence),
                        bbox=line.bbox,
                        page_number=page.page_number,
                    )
                )
            continue
        for tok in page.tokens:
            text = (tok.text or "").strip()
            if not text:
                continue
            out.append(
                OcrRegion(
                    text=text,
                    confidence=float(tok.confidence),
                    bbox=tok.bbox,
                    page_number=page.page_number,
                )
            )
    return out
