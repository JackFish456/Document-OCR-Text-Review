"""Google Cloud Vision OCR (DOCUMENT_TEXT_DETECTION) -> :class:`OCRResult`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

from app.core.logging import get_logger
from app.ocr.base import BBoxInt, OCRProvider, OCRResult
from app.ocr.document_provider import OCRDependencyError, OCRImageLoadError, OCRProviderError
from app.ocr.normalizer import normalize_ocr_result

logger = get_logger(__name__)

try:
    from google.cloud import vision
except ImportError:
    vision = None  # type: ignore[assignment]


def _require_vision() -> Any:
    if vision is None:
        raise OCRDependencyError(
            "google-cloud-vision is required for GoogleVisionOCRProvider. "
            "Install the optional extra (e.g. pip install drawing-compare[ocr-google])."
        )
    return vision


def _clip_confidence(value: float) -> float:
    c = float(value)
    if c > 1.0 and c <= 100.0:
        c = c / 100.0
    return max(0.0, min(1.0, c))


def _dims_from_image_bytes(data: bytes) -> tuple[int, int]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return (1, 1)
    h, w = img.shape[:2]
    return (int(w), int(h))


def _poly_to_bbox(
    poly: Any,
    page_w: int,
    page_h: int,
) -> BBoxInt:
    """Convert Vision ``BoundingPoly`` to pixel [x1,y1,x2,y2] (same orientation as local OCR)."""
    if poly is None:
        return (0, 0, 1, 1)
    if getattr(poly, "normalized_vertices", None):
        xs = [float(v.x) * page_w for v in poly.normalized_vertices]
        ys = [float(v.y) * page_h for v in poly.normalized_vertices]
    elif getattr(poly, "vertices", None):
        xs = [float(v.x) for v in poly.vertices]
        ys = [float(v.y) for v in poly.vertices]
    else:
        return (0, 0, 1, 1)
    if not xs or not ys:
        return (0, 0, 1, 1)
    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))
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


def _word_text(word: Any) -> str:
    return "".join(s.text for s in word.symbols)


def _word_confidence(word: Any) -> float:
    if getattr(word, "confidence", None) and float(word.confidence) > 0.0:
        return _clip_confidence(float(word.confidence))
    confs = [float(s.confidence) for s in word.symbols if getattr(s, "confidence", None)]
    confs = [c for c in confs if c > 0.0]
    if confs:
        return _clip_confidence(sum(confs) / len(confs))
    return 1.0


def _paragraph_to_line(paragraph: Any, page_w: int, page_h: int) -> OCRResult.Line | None:
    tokens: list[OCRResult.Token] = []
    word_confs: list[float] = []
    for word in paragraph.words:
        raw = _word_text(word)
        text = raw.strip()
        if not text:
            continue
        bbox = _poly_to_bbox(word.bounding_box, page_w, page_h)
        conf = _word_confidence(word)
        word_confs.append(conf)
        tokens.append(OCRResult.Token(text=text, bbox=bbox, confidence=conf))
    if not tokens:
        return None
    line_text = " ".join(t.text for t in tokens)
    pbox = getattr(paragraph, "bounding_box", None)
    if pbox is not None and (
        getattr(pbox, "vertices", None) or getattr(pbox, "normalized_vertices", None)
    ):
        line_bbox = _poly_to_bbox(pbox, page_w, page_h)
    else:
        line_bbox = _union_bboxes([t.bbox for t in tokens])
    line_conf = (
        sum(word_confs) / len(word_confs) if word_confs else 1.0
    )
    return OCRResult.Line(
        text=line_text,
        bbox=line_bbox,
        confidence=_clip_confidence(line_conf),
        tokens=tokens,
    )


def _parse_vision_page(
    annotation: Any,
    *,
    page_number: int,
    fallback_w: int,
    fallback_h: int,
) -> OCRResult.Page:
    if not annotation or not annotation.pages:
        fw = max(1, fallback_w)
        fh = max(1, fallback_h)
        return OCRResult.Page(page_number=page_number, width=fw, height=fh, lines=[])

    vp = annotation.pages[0]
    pw = int(vp.width) if vp.width else fallback_w
    ph = int(vp.height) if vp.height else fallback_h
    pw = max(1, pw)
    ph = max(1, ph)

    lines: list[OCRResult.Line] = []
    for block in vp.blocks:
        for paragraph in block.paragraphs:
            line = _paragraph_to_line(paragraph, pw, ph)
            if line is not None:
                lines.append(line)

    return OCRResult.Page(page_number=page_number, width=pw, height=ph, lines=lines)


def _iter_vision_page_payloads(path: Path) -> list[tuple[int, bytes, int, int]]:
    """Per-page image bytes: render PDF/TIFF; pass single images through unchanged."""
    suffix = path.suffix.lower()
    if suffix == ".pdf" or suffix in (".tif", ".tiff"):
        doc = fitz.open(path)
        try:
            out: list[tuple[int, bytes, int, int]] = []
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            for i in range(len(doc)):
                page = doc[i]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                png_bytes = pix.tobytes("png")
                out.append((i + 1, png_bytes, int(pix.width), int(pix.height)))
            return out
        finally:
            doc.close()

    data = path.read_bytes()
    w, h = _dims_from_image_bytes(data)
    return [(1, data, w, h)]


class GoogleVisionOCRProvider(OCRProvider):
    """Runs Cloud Vision ``DOCUMENT_TEXT_DETECTION`` and maps results to :class:`OCRResult`.

    Authentication:
    - Set ``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account JSON path, **or**
    - Pass ``credentials_path`` explicitly, **or**
    - Rely on Application Default Credentials (e.g. workload identity).

    PDFs and multi-page TIFFs are rasterized per page with PyMuPDF so each page uses the same
    pixel-coordinate convention as single-image OCR (aligned with local providers).
    """

    def __init__(
        self,
        *,
        credentials_path: str | None = None,
    ) -> None:
        self._credentials_path = credentials_path
        self._client: Any | None = None

    def _resolve_credentials_path(self) -> str | None:
        if self._credentials_path and str(self._credentials_path).strip():
            return str(Path(self._credentials_path).expanduser())
        env_path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        if env_path:
            return env_path
        from app.core.config import get_settings

        cfg = get_settings().google_application_credentials
        if cfg and str(cfg).strip():
            return str(Path(cfg).expanduser())
        return None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        vis = _require_vision()
        path = self._resolve_credentials_path()
        if path:
            p = Path(path).expanduser().resolve()
            if not p.is_file():
                raise OCRImageLoadError(
                    f"GOOGLE_APPLICATION_CREDENTIALS path is not a file: {path!s}"
                )
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(str(p))
            self._client = vis.ImageAnnotatorClient(credentials=creds)
        else:
            self._client = vis.ImageAnnotatorClient()
        return self._client

    def extract_raw(self, document_path: str) -> OCRResult:
        """Vision API → page :class:`OCRResult` before :func:`normalize_ocr_result`."""
        return self._extract_pages_before_normalize(document_path)

    def extract(self, document_path: str) -> OCRResult:
        return normalize_ocr_result(self.extract_raw(document_path))

    def _extract_pages_before_normalize(self, document_path: str) -> OCRResult:
        vis = _require_vision()
        path = Path(document_path).expanduser().resolve()
        if not path.is_file():
            raise OCRImageLoadError(f"Not a file or missing: {document_path!r}")

        payloads = _iter_vision_page_payloads(path)
        pages_out: list[OCRResult.Page] = []
        client = self._get_client()

        for page_number, img_bytes, fb_w, fb_h in payloads:
            image = vis.Image(content=img_bytes)
            response = client.document_text_detection(image=image)
            err = getattr(response, "error", None)
            if err is not None and err.code != 0:
                raise OCRProviderError(
                    f"Google Vision DOCUMENT_TEXT_DETECTION failed: {err.message} (code {err.code})"
                )
            ann = response.full_text_annotation
            pages_out.append(
                _parse_vision_page(
                    ann,
                    page_number=page_number,
                    fallback_w=fb_w,
                    fallback_h=fb_h,
                )
            )

        logger.info(
            "Google Vision OCR finished",
            extra={"path": str(path), "pages": len(pages_out)},
        )
        return OCRResult(pages=pages_out)
