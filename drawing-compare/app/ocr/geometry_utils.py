"""Shared geometry helpers for OCR bounding boxes."""

from __future__ import annotations

from app.models.ocr import BoundingBox

# Smallest axis-aligned extent so :class:`BoundingBox` validators accept degenerate OCR boxes.
_MIN_EXTENT = 1e-3


def bbox_from_xywh(x: float, y: float, width: float, height: float) -> BoundingBox:
    """Build a validated axis-aligned box from top-left and size (pixel coordinates)."""
    w = max(float(width), _MIN_EXTENT)
    h = max(float(height), _MIN_EXTENT)
    return BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h)


def bbox_from_quad(points: list[tuple[float, float]]) -> BoundingBox:
    """Axis-aligned envelope of a quadrilateral (e.g. PaddleOCR rotated box)."""
    if len(points) < 2:
        raise ValueError("quad requires at least two points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return bbox_from_xywh(x1, y1, max(x2 - x1, _MIN_EXTENT), max(y2 - y1, _MIN_EXTENT))


def clamp_confidence(value: float | int | None, *, scale_0_100: bool = False) -> float:
    """Normalize confidence to 0..1."""
    if value is None:
        return 0.0
    v = float(value)
    if scale_0_100:
        if v < 0:
            return 0.0
        return max(0.0, min(1.0, v / 100.0))
    return max(0.0, min(1.0, v))
