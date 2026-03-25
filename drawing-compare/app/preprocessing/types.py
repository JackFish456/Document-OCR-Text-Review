"""Structured preprocessing outputs (paths + in-memory rasters for OCR)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

BGRImage = NDArray[np.uint8]


class PreprocessArtifacts(BaseModel):
    """On-disk artifacts for audit and debugging."""

    original_png: Path | None = None
    processed_png: Path | None = None
    source_uri: str = ""


@dataclass
class PreprocessedTile:
    """One tile crop in page pixel space (after deskew / full pipeline)."""

    row: int
    col: int
    offset_x: int
    offset_y: int
    width: int
    height: int
    processed_bgr: BGRImage


@dataclass
class PreprocessPageResult:
    """Single page/sheet result: preserve originals, expose OCR-ready rasters."""

    page_number: int
    source_path: str
    width: int
    height: int
    original_bgr: BGRImage
    processed_bgr: BGRImage
    deskew_angle_deg: float
    tiles: list[PreprocessedTile] = field(default_factory=list)
    artifacts: PreprocessArtifacts | None = None

    def primary_ocr_image(self) -> BGRImage:
        """Full-page OCR input (respects max_long_edge internally in pipeline)."""
        return self.processed_bgr
