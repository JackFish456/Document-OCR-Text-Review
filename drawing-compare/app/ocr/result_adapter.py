"""Map :class:`OCRResult` to legacy :class:`OCRDocument` for parsing and region extraction."""

from __future__ import annotations

from app.models.ocr import BoundingBox, OCRDocument, OCRLine, OCRPage, OCRToken
from app.ocr.base import OCRProvider, OCRResult


def _bbox_from_tuple(b: tuple[int, int, int, int]) -> BoundingBox:
    x1, y1, x2, y2 = b
    return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def ocr_provider_document_label(provider: OCRProvider) -> str:
    """``provider_name`` for :class:`OCRDocument` (inner backend name when local)."""
    from app.ocr.google_provider import GoogleVisionOCRProvider
    from app.ocr.local_provider import LocalOCRProvider

    if isinstance(provider, LocalOCRProvider):
        return provider.inner.provider_name
    if isinstance(provider, GoogleVisionOCRProvider):
        return "google_vision"
    return "ocr_result"


def ocr_result_to_document(
    result: OCRResult,
    *,
    document_id: str,
    source_path: str = "",
    provider_name: str = "",
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> OCRDocument:
    """Convert normalized :class:`OCRResult` into the legacy hierarchical document model."""
    pages_out: list[OCRPage] = []
    for page in result.pages:
        lines_out: list[OCRLine] = []
        for li, line in enumerate(page.lines):
            line_id = f"p{page.page_number}-l{li}"
            tokens_out: list[OCRToken] = []
            for tok in line.tokens:
                tokens_out.append(
                    OCRToken(
                        text=tok.text,
                        bbox=_bbox_from_tuple(tok.bbox),
                        confidence=tok.confidence,
                        line_id=line_id,
                    )
                )
            oline = OCRLine(
                id=line_id,
                text=line.text,
                bbox=_bbox_from_tuple(line.bbox),
                confidence=line.confidence,
                tokens=tokens_out,
            )
            lines_out.append(oline.with_token_line_ids())
        pages_out.append(
            OCRPage(
                page_number=page.page_number,
                width=float(page.width),
                height=float(page.height),
                lines=lines_out,
            )
        )
    return OCRDocument(
        document_id=document_id,
        source_path=source_path,
        provider_name=provider_name,
        pages=pages_out,
        metadata=dict(metadata or {}),
    )
