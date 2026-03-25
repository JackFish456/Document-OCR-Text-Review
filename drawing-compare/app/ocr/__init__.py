"""OCR provider abstraction and registrations."""

from app.ocr.base import OcrImageInput as OcrImageInput
from app.ocr.base import OcrProvider as OcrProvider
from app.ocr.document_provider import (
    CloudOCRNotImplementedError,
    OCRDependencyError,
    OCRError,
    OCRImageLoadError,
    OCRProvider,
    OCRProviderError,
)
from app.ocr.factory import get_document_ocr_provider, get_ocr_provider

__all__ = [
    "CloudOCRNotImplementedError",
    "OCRError",
    "OCRDependencyError",
    "OCRImageLoadError",
    "OCRImageInput",
    "OCRProvider",
    "OCRProviderError",
    "OcrProvider",
    "get_document_ocr_provider",
    "get_ocr_provider",
]
