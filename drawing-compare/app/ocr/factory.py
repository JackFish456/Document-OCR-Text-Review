"""Resolve OCR provider from settings."""

from app.core.config import Settings
from app.ocr.base import OCRProvider, OcrProvider
from app.ocr.document_adapter import DocumentProviderAdapter
from app.ocr.document_provider import OCRProvider as DocumentOCRProvider
from app.ocr.local_provider import LocalOCRProvider
from app.ocr.stub_document_provider import StubDocumentOCRProvider
from app.ocr.stub_provider import EchoOcrProvider, StubOcrProvider

_DOCUMENT_PROVIDER_ALIASES: dict[str, str] = {
    "stub": "stub_document",
    "paddle": "paddleocr",
    "azure": "azure_document_intelligence",
    "google": "google_vision",
    "windows": "windows_ocr",
}


def get_ocr_provider(settings: Settings) -> OcrProvider:
    key = settings.ocr_provider.lower().strip()
    if key == "stub":
        return StubOcrProvider()
    if key == "echo":
        return EchoOcrProvider()
    # Bridge document providers (file-based) into ndarray pipeline via temp image adapter.
    bridge_settings = settings.model_copy(update={"document_ocr_provider": key})
    doc_provider = get_document_ocr_provider(bridge_settings)
    scratch = settings.data_dir / "tmp" / "ocr_adapter"
    return DocumentProviderAdapter(doc_provider, scratch_dir=scratch)


def get_local_ocr_provider(settings: Settings) -> LocalOCRProvider:
    """Always the local legacy stack as :class:`OCRResult` (ignores ``ocr_result_backend``)."""
    return LocalOCRProvider(inner=get_document_ocr_provider(settings))


def get_ocr_result_provider(settings: Settings) -> OCRProvider:
    """Return :class:`OCRResult` provider per ``ocr_result_backend`` (local or google)."""
    if settings.ocr_result_backend == "google":
        from app.ocr.google_provider import GoogleVisionOCRProvider

        return GoogleVisionOCRProvider(credentials_path=settings.google_application_credentials)
    return LocalOCRProvider(inner=get_document_ocr_provider(settings))


def get_document_ocr_provider(settings: Settings) -> DocumentOCRProvider:
    """Instantiate the configured file-based OCR backend (see ``document_ocr_provider``)."""
    raw = settings.document_ocr_provider.lower().strip()
    key = _DOCUMENT_PROVIDER_ALIASES.get(raw, raw)

    if key in ("stub_document",):
        return StubDocumentOCRProvider()
    if key == "tesseract":
        from app.ocr.providers.tesseract import TesseractOCRProvider

        return TesseractOCRProvider(
            lang=settings.tesseract_lang,
            config=settings.tesseract_config,
            psm=settings.tesseract_psm,
        )
    if key == "paddleocr":
        from app.ocr.providers.paddle import PaddleOCRProvider

        return PaddleOCRProvider(
            lang=settings.paddle_lang,
            use_angle_cls=settings.paddle_use_angle_cls,
            use_gpu=settings.paddle_use_gpu,
        )
    if key == "azure_document_intelligence":
        from app.ocr.providers.azure import AzureDocumentIntelligenceProvider

        return AzureDocumentIntelligenceProvider()
    if key == "google_vision":
        from app.ocr.providers.google_vision import GoogleVisionOCRProvider

        return GoogleVisionOCRProvider()
    if key == "windows_ocr":
        from app.ocr.providers.windows_ocr import WindowsOCRProvider

        return WindowsOCRProvider()

    raise ValueError(
        f"Unknown document OCR provider: {settings.document_ocr_provider!r} "
        f"(resolved: {key!r}). "
        "Expected stub_document, tesseract, paddleocr (alias: paddle), "
        "windows_ocr (alias: windows), azure_document_intelligence (alias: azure), "
        "google_vision (alias: google)."
    )
