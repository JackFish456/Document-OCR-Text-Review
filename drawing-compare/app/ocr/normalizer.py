"""Post-process :class:`OCRResult` for consistent text, geometry, and confidence."""

from __future__ import annotations

import re

from app.ocr.base import BBoxInt, OCRResult


def _normalize_text(s: str) -> str:
    """Strip and collapse internal whitespace to a single space."""
    t = (s or "").strip()
    return re.sub(r"\s+", " ", t)


def _normalize_confidence(value: float) -> float:
    c = float(value)
    if c > 1.0 and c <= 100.0:
        c = c / 100.0
    return max(0.0, min(1.0, c))


def _normalize_bbox(bbox: BBoxInt) -> BBoxInt:
    """Integer axis-aligned box with positive extent (matches local OCR conventions)."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def _union_bboxes(boxes: list[BBoxInt]) -> BBoxInt:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def _normalize_token(tok: OCRResult.Token) -> OCRResult.Token | None:
    text = _normalize_text(tok.text)
    if not text:
        return None
    return OCRResult.Token(
        text=text,
        bbox=_normalize_bbox(tok.bbox),
        confidence=_normalize_confidence(tok.confidence),
    )


def _normalize_line(line: OCRResult.Line) -> OCRResult.Line | None:
    tokens: list[OCRResult.Token] = []
    for t in line.tokens:
        nt = _normalize_token(t)
        if nt is not None:
            tokens.append(nt)

    if tokens:
        line_text = _normalize_text(" ".join(t.text for t in tokens))
        confs = [t.confidence for t in tokens]
        line_conf = _normalize_confidence(sum(confs) / len(confs))
        line_bbox = _union_bboxes([t.bbox for t in tokens])
    else:
        line_text = _normalize_text(line.text)
        if not line_text:
            return None
        line_conf = _normalize_confidence(line.confidence)
        line_bbox = _normalize_bbox(line.bbox)

    return OCRResult.Line(
        text=line_text,
        bbox=line_bbox,
        confidence=line_conf,
        tokens=tokens,
    )


def _normalize_page(page: OCRResult.Page, *, page_number: int) -> OCRResult.Page:
    lines_out: list[OCRResult.Line] = []
    for line in page.lines:
        nl = _normalize_line(line)
        if nl is not None:
            lines_out.append(nl)
    w = max(1, int(round(page.width)))
    h = max(1, int(round(page.height)))
    return OCRResult.Page(
        page_number=page_number,
        width=w,
        height=h,
        lines=lines_out,
    )


def normalize_ocr_result(result: OCRResult) -> OCRResult:
    """Return a copy with stable layout, text cleanup, and confidence/bbox consistency.

    - Text: stripped; runs of whitespace collapsed to a single space.
    - Bboxes: integer ``[x1,y1,x2,y2]`` with minimum 1px width/height.
    - Pages: sorted by ``page_number``, then renumbered ``1..n`` in that order.
    - Lines: tokens first; line text rebuilt from tokens; empty lines dropped.
    - Confidence: clipped to ``[0, 1]``; values in ``(1, 100]`` treated as percent.
    """
    pages_sorted = sorted(result.pages, key=lambda p: p.page_number)
    out_pages: list[OCRResult.Page] = []
    for i, page in enumerate(pages_sorted):
        out_pages.append(_normalize_page(page, page_number=i + 1))
    return OCRResult(pages=out_pages)
