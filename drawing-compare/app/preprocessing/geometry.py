"""Shared geometric helpers for preprocessing."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.preprocessing.types import BGRImage


def downscale_long_edge(image_bgr: BGRImage, max_dim: int) -> BGRImage:
    """Resize so longest edge is at most max_dim (no-op if max_dim is 0 or image smaller)."""
    if max_dim <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_bgr
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cast(BGRImage, cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA))


def rotate_bound(image: NDArray[np.uint8], angle_deg: float) -> NDArray[np.uint8]:
    """Rotate around image center, expanding bounds so nothing is cropped."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    nw = int(round(h * sin + w * cos))
    nh = int(round(h * cos + w * sin))
    m[0, 2] += (nw / 2) - center[0]
    m[1, 2] += (nh / 2) - center[1]
    return cast(
        NDArray[np.uint8],
        cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
    )
