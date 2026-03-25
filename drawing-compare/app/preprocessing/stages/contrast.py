"""CLAHE contrast enhancement on grayscale."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]


def apply_clahe(gray: GrayImage, *, clip_limit: float, tile_size: int) -> GrayImage:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return cast(GrayImage, clahe.apply(gray))
