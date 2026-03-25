"""Noise reduction on grayscale (bilateral default for large drawings)."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.preprocessing.settings import PreprocessConfig

GrayImage = NDArray[np.uint8]


def denoise_gray(
    gray: GrayImage,
    cfg: PreprocessConfig,
    *,
    method: Literal["bilateral", "nlmeans"] | None = None,
) -> GrayImage:
    use: Literal["bilateral", "nlmeans"] = method or cfg.denoise_method
    if use == "bilateral":
        if cfg.bilateral_d % 2 == 0:
            d = cfg.bilateral_d + 1
        else:
            d = cfg.bilateral_d
        return cast(
            GrayImage,
            cv2.bilateralFilter(
                gray,
                d,
                cfg.bilateral_sigma_color,
                cfg.bilateral_sigma_space,
            ),
        )
    return cast(GrayImage, cv2.fastNlMeansDenoising(gray, h=cfg.nlmeans_h))
