"""Adaptive thresholding for line-dominant drawings."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]


def adaptive_threshold(
    gray: GrayImage,
    *,
    block_size: int,
    c: int,
    invert: bool,
) -> GrayImage:
    """Return single-channel image (binary 0/255)."""
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cast(
        GrayImage,
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            mode,
            block_size,
            c,
        ),
    )
