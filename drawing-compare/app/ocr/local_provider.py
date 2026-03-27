"""Local document OCR as :class:`OCRResult` via legacy ``OCRDocument`` providers."""

from __future__ import annotations

from app.models.ocr import BoundingBox, OCRDocument, OCRLine, OCRPage, OCRToken
from app.ocr.base import BBoxInt, OCRProvider, OCRResult
from app.ocr.document_provider import OCRProvider as DocumentOCRProvider
from app.ocr.normalizer import normalize_ocr_result


def _clip_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox_to_page_pixels(bbox: BoundingBox) -> BBoxInt:
    """Map legacy float geometry to integer page space (same origin and axis directions)."""
    x1 = int(round(bbox.x1))
    y1 = int(round(bbox.y1))
    x2 = int(round(bbox.x2))
    y2 = int(round(bbox.y2))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def _effective_line_confidence(line: OCRLine) -> float:
    if line.tokens:
        mean_t = sum(t.confidence for t in line.tokens) / len(line.tokens)
        if line.confidence > 0.0:
            return _clip_confidence(line.confidence)
        return _clip_confidence(mean_t)
    if line.confidence > 0.0:
        return _clip_confidence(line.confidence)
    return 1.0


def _token_to_normalized(t: OCRToken) -> OCRResult.Token | None:
    text = (t.text or "").strip()
    if not text:
        return None
    return OCRResult.Token(
        text=text,
        bbox=_bbox_to_page_pixels(t.bbox),
        confidence=_clip_confidence(t.confidence),
    )


def _line_to_normalized(line: OCRLine) -> OCRResult.Line:
    tokens: list[OCRResult.Token] = []
    for t in line.tokens:
        nt = _token_to_normalized(t)
        if nt is not None:
            tokens.append(nt)
    return OCRResult.Line(
        text=line.text or "",
        bbox=_bbox_to_page_pixels(line.bbox),
        confidence=_effective_line_confidence(line),
        tokens=tokens,
    )


def _page_to_normalized(page: OCRPage) -> OCRResult.Page:
    w = max(1, int(round(page.width)))
    h = max(1, int(round(page.height)))
    return OCRResult.Page(
        page_number=page.page_number,
        width=w,
        height=h,
        lines=[_line_to_normalized(line) for line in page.lines],
    )


def ocr_document_to_result(doc: OCRDocument) -> OCRResult:
    """Convert a legacy :class:`OCRDocument` into normalized :class:`OCRResult`."""
    return OCRResult(pages=[_page_to_normalized(p) for p in doc.pages])


class LocalOCRProvider(OCRProvider):
    """Runs the configured in-app document OCR backend and returns :class:`OCRResult`."""

    def __init__(self, inner: DocumentOCRProvider) -> None:
        self._inner = inner

    @property
    def inner(self) -> DocumentOCRProvider:
        return self._inner

    def extract_raw(self, document_path: str) -> OCRResult:
        """Return :class:`OCRResult` before :func:`normalize_ocr_result` (legacy → page space)."""
        doc = self._inner.extract(document_path)
        return ocr_document_to_result(doc)

    def extract(self, document_path: str) -> OCRResult:
        return normalize_ocr_result(self.extract_raw(document_path))
