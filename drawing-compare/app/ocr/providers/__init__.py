"""Concrete OCR provider implementations."""

from app.ocr.providers.azure import AzureDocumentIntelligenceProvider
from app.ocr.providers.google_vision import GoogleVisionOCRProvider
from app.ocr.providers.paddle import PaddleOCRProvider
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.providers.windows_ocr import WindowsOCRProvider

__all__ = [
    "AzureDocumentIntelligenceProvider",
    "GoogleVisionOCRProvider",
    "PaddleOCRProvider",
    "TesseractOCRProvider",
    "WindowsOCRProvider",
]
