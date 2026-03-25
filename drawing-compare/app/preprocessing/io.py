"""Load raster images and rasterize PDF pages for preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import fitz
import numpy as np
from numpy.typing import NDArray

BGRImage = NDArray[np.uint8]


def load_image_bgr(path: Path | str) -> BGRImage:
    """Load an image file as BGR uint8."""
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        msg = f"Could not read image: {path}"
        raise FileNotFoundError(msg)
    return cast(BGRImage, data)


def cap_longest_edge(image_bgr: BGRImage, max_dim: int) -> BGRImage:
    """Downscale if longest edge exceeds max_dim (0 = no cap)."""
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


def pdf_to_bgr_pages(
    path: Path | str,
    *,
    render_dpi: float,
    max_raster_dimension: int,
) -> list[BGRImage]:
    """Rasterize each PDF page to BGR, applying optional max dimension cap."""
    doc = fitz.open(str(path))
    try:
        zoom = render_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pages: list[BGRImage] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            h, w = pix.height, pix.width
            n = pix.n
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, n)
            if n == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif n == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                msg = f"Unexpected PDF pixmap channels: {n}"
                raise ValueError(msg)
            pages.append(cap_longest_edge(cast(BGRImage, bgr), max_raster_dimension))
        return pages
    finally:
        doc.close()


def is_pdf_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() == ".pdf"
