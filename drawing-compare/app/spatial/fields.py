"""Spatial measures between :class:`~app.models.extraction.ExtractedField` rows (same-page aware)."""

from __future__ import annotations

import math

from app.models.extraction import ExtractedField
from app.models.ocr import BoundingBox


def center_point(bbox: BoundingBox) -> tuple[float, float]:
    """Return ``(cx, cy)`` for the box (delegates to :meth:`~app.models.ocr.BoundingBox.center`)."""
    return bbox.center()


def distance_between_boxes(a: BoundingBox, b: BoundingBox) -> float:
    """Euclidean distance between box centers."""
    ca, cb = a.center(), b.center()
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def distance_between_fields(a: ExtractedField, b: ExtractedField) -> float:
    """Distance between field centers; ``inf`` on different pages (no cross-page metric yet)."""
    if a.page_number != b.page_number:
        return float("inf")
    return distance_between_boxes(a.bbox, b.bbox)


def boxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    """True when the two boxes have positive-area overlap."""
    return a.intersects(b)


def overlap_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union for two boxes."""
    return a.iou(b)


def overlap_iou_fields(a: ExtractedField, b: ExtractedField) -> float:
    """IoU when ``same page``; otherwise ``0.0`` (treat as non-overlapping across pages)."""
    if a.page_number != b.page_number:
        return 0.0
    return a.bbox.iou(b.bbox)
