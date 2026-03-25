"""Construct :class:`OCRDocument` / :class:`OCRPage` with consistent token–line geometry."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.models.ocr import BoundingBox, OCRDocument, OCRLine, OCRPage, OCRToken
from app.ocr.document_provider import OCRImageLoadError
from app.ocr.geometry_utils import bbox_from_xywh, clamp_confidence


def read_image_bgr(path: Path) -> NDArray[np.uint8]:
    """Load an image as BGR uint8."""
    arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise OCRImageLoadError(f"OpenCV could not decode image: {path}")
    return cast(NDArray[np.uint8], arr)


def image_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for the first page / raster."""
    bgr = read_image_bgr(path)
    h, w = bgr.shape[:2]
    return int(w), int(h)


def build_document_shell(
    *,
    document_id: str,
    source_path: str,
    provider_name: str,
    width: float,
    height: float,
    lines: list[OCRLine],
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> OCRDocument:
    """Attach line ids to tokens and assemble a single-page document."""
    stamped_lines: list[OCRLine] = []
    for line in lines:
        stamped = line.with_token_line_ids()
        stamped_lines.append(stamped)
    page = OCRPage(
        page_number=1,
        width=width,
        height=height,
        lines=stamped_lines,
    )
    return OCRDocument(
        document_id=document_id,
        source_path=source_path,
        provider_name=provider_name,
        pages=[page],
        metadata=dict(metadata or {}),
    )


def line_id_for(page_no: int, idx: int) -> str:
    return f"p{page_no}-l{idx}"


def allocate_token_boxes_along_line(
    line_text: str,
    line_bbox: BoundingBox,
    *,
    base_confidence: float,
    line_id: str,
) -> list[OCRToken]:
    """Split a line on whitespace and assign proportional horizontal slices of ``line_bbox``."""
    parts = [w for w in line_text.split() if w]
    if not parts:
        return []

    line_w = line_bbox.width()
    x_cursor = line_bbox.x1
    tokens: list[OCRToken] = []
    # Each word gets width proportional to its character count (simple, layout-agnostic).
    total_weight = sum(max(len(p), 1) for p in parts)
    for word in parts:
        weight = max(len(word), 1)
        slice_w = line_w * (weight / total_weight)
        tok_box = bbox_from_xywh(x_cursor, line_bbox.y1, slice_w, line_bbox.height())
        tokens.append(
            OCRToken(
                text=word,
                confidence=clamp_confidence(base_confidence),
                bbox=tok_box,
                line_id=line_id,
            )
        )
        x_cursor += slice_w
    return tokens


def make_line_from_tokens(
    tokens: list[OCRToken],
    *,
    line_id: str,
    page_no: int,
    line_idx: int,
) -> OCRLine:
    """Wrap tokens in a line; bbox is the union of token boxes."""
    if not tokens:
        raise ValueError("tokens required")
    x1 = min(t.bbox.x1 for t in tokens)
    y1 = min(t.bbox.y1 for t in tokens)
    x2 = max(t.bbox.x2 for t in tokens)
    y2 = max(t.bbox.y2 for t in tokens)
    union = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    confs = [t.confidence for t in tokens]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    text = " ".join(t.text for t in tokens)
    lid = line_id or line_id_for(page_no, line_idx)
    return OCRLine(
        id=lid,
        text=text,
        confidence=mean_conf,
        bbox=union,
        tokens=[t.model_copy(update={"line_id": lid}) for t in tokens],
    )
