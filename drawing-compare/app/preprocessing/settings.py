"""Preprocessing pipeline configuration (nested under app `Settings`)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class PreprocessConfig(BaseModel):
    """Tunable GA/engineering drawing preprocessing. Env: DRAWING_COMPARE_PREPROCESS__*."""

    # --- Input / raster ---
    pdf_render_dpi: float = Field(default=300.0, ge=72.0, le=600.0)
    max_raster_dimension: int = Field(
        default=8192,
        ge=512,
        description="Cap longest edge after PDF render (downscale if larger)",
    )

    # --- Stage toggles ---
    enable_grayscale: bool = True
    enable_clahe: bool = True
    clahe_clip_limit: float = Field(default=2.0, gt=0.0, le=10.0)
    clahe_tile_size: int = Field(default=8, ge=2, le=64)

    enable_denoise: bool = True
    denoise_method: Literal["bilateral", "nlmeans"] = "bilateral"
    bilateral_d: int = Field(default=7, ge=1, le=15)
    bilateral_sigma_color: float = Field(default=75.0, gt=0.0)
    bilateral_sigma_space: float = Field(default=75.0, gt=0.0)
    nlmeans_h: float = Field(default=10.0, gt=0.0)

    enable_adaptive_threshold: bool = True
    adaptive_block_size: int = Field(default=35, ge=3, description="Must be odd")
    adaptive_c: int = Field(default=10, ge=0)
    adaptive_invert: bool = False

    enable_deskew: bool = True
    deskew_max_deg: float = Field(default=15.0, ge=0.0, le=45.0)

    # --- Output geometry ---
    max_long_edge_output: int = Field(
        default=4096,
        ge=0,
        description="Max longest edge before OCR; 0 disables downscaling",
    )

    # --- Tiling ---
    enable_tiling: bool = False
    tile_if_min_side_gt: int = Field(
        default=4096,
        ge=32,
        description="When enable_tiling, also tile if min(width,height) exceeds this",
    )
    tile_size: int = Field(default=2048, ge=32)
    tile_overlap_ratio: float = Field(default=0.1, ge=0.0, lt=0.5)

    # --- Artifacts ---
    write_intermediates: bool = Field(
        default=True,
        description="When output_dir is set, write PNGs for original and processed",
    )
    artifact_format: Literal["png"] = "png"

    @field_validator("adaptive_block_size")
    @classmethod
    def odd_block_size(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("adaptive_block_size must be odd")
        return v

    @model_validator(mode="after")
    def tile_overlap_sane(self) -> Self:
        if self.tile_overlap_ratio * self.tile_size < 1 and self.enable_tiling:
            raise ValueError("tile_overlap_ratio * tile_size must be >= 1 pixel")
        return self
