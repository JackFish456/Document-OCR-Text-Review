"""Google Cloud Vision OCR — interface stub (client integration pending)."""

from __future__ import annotations

from app.models.ocr import OCRDocument
from app.ocr.document_builders import image_size
from app.ocr.document_provider import CloudOCRNotImplementedError, OCRProvider


class GoogleVisionOCRProvider(OCRProvider):
    """Reserved provider; call :meth:`extract` after wiring ``google-cloud-vision``."""

    def __init__(self, *, credentials_path: str | None = None) -> None:
        self._credentials_path = credentials_path

    @property
    def provider_name(self) -> str:
        return "google_vision"

    def extract(self, image_path: str) -> OCRDocument:
        path = self._validate_path(image_path)
        self._log_extract_start(str(path))
        _ = image_size(path)
        raise CloudOCRNotImplementedError(
            "Google Cloud Vision OCR is not implemented. "
            "Add dependency google-cloud-vision, authenticate (ADC or JSON key), "
            "and map document_text_detection / text_annotation pages into OCRDocument."
        )
