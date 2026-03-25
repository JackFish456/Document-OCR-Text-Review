"""RGB/BGR to single-channel grayscale."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.preprocessing.types import BGRImage

GrayImage = NDArray[np.uint8]


def to_gray(image_bgr: BGRImage) -> GrayImage:
    return cast(GrayImage, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
