"""Skew estimation for line-heavy drawings."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]


def estimate_skew_deg(gray: GrayImage) -> float:
    """Estimate dominant skew (degrees); positive ≈ clockwise rotation of content."""
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords_yx = np.column_stack(np.where(bw > 0))
    if coords_yx.shape[0] < 500:
        return 0.0
    coords_xy = np.column_stack((coords_yx[:, 1], coords_yx[:, 0])).astype(np.float32)
    rect = cv2.minAreaRect(coords_xy)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return float(angle)
