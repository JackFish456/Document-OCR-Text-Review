"""Azure AI Document Intelligence — interface stub (SDK integration pending)."""

from __future__ import annotations

from app.models.ocr import OCRDocument
from app.ocr.document_builders import image_size
from app.ocr.document_provider import CloudOCRNotImplementedError, OCRProvider


class AzureDocumentIntelligenceProvider(OCRProvider):
    """Reserved provider; call :meth:`extract` after wiring ``azure-ai-documentintelligence``."""

    def __init__(self, *, endpoint: str | None = None, api_key: str | None = None) -> None:
        self._endpoint = endpoint
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "azure_document_intelligence"

    def extract(self, image_path: str) -> OCRDocument:
        path = self._validate_path(image_path)
        self._log_extract_start(str(path))
        _ = image_size(path)  # fail fast on corrupt inputs; normalized output is still TODO
        raise CloudOCRNotImplementedError(
            "Azure Document Intelligence is not implemented. "
            "Add dependency azure-ai-documentintelligence, configure endpoint and key, "
            "and map analyze result pages/lines/spans into OCRDocument."
        )
