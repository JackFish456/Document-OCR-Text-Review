"""GA drawing preprocessing: PDF/image ingest, classical CV stages, tiling."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.config import Settings
from app.preprocessing.deskew import estimate_skew_deg
from app.preprocessing.geometry import downscale_long_edge, rotate_bound
from app.preprocessing.io import is_pdf_path, load_image_bgr, pdf_to_bgr_pages
from app.preprocessing.settings import PreprocessConfig
from app.preprocessing.stages.contrast import apply_clahe
from app.preprocessing.stages.denoise import denoise_gray
from app.preprocessing.stages.grayscale import to_gray
from app.preprocessing.stages.threshold import adaptive_threshold
from app.preprocessing.stages.tiling import build_tiles, should_tile
from app.preprocessing.types import BGRImage, PreprocessArtifacts, PreprocessedTile, PreprocessPageResult

GrayImage = NDArray[np.uint8]


def _write_png(path: Path, image: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        msg = f"Failed to write image: {path}"
        raise OSError(msg)


def _process_single_page(
    original_bgr: BGRImage,
    *,
    source_path: str,
    page_number: int,
    cfg: PreprocessConfig,
    output_dir: Path | None,
    base_name: str,
) -> PreprocessPageResult:
    """Run configured stages; `original_bgr` is unchanged raster (pre deskew)."""
    color_work = original_bgr.copy()
    deskew_angle = 0.0

    if cfg.enable_grayscale:
        gray: GrayImage = to_gray(color_work)
        if cfg.enable_clahe:
            gray = apply_clahe(gray, clip_limit=cfg.clahe_clip_limit, tile_size=cfg.clahe_tile_size)
        if cfg.enable_deskew:
            raw_angle = estimate_skew_deg(gray)
            clipped = float(np.clip(raw_angle, -cfg.deskew_max_deg, cfg.deskew_max_deg))
            if abs(clipped) >= 0.05:
                deskew_angle = -clipped
                color_work = rotate_bound(color_work, deskew_angle)
                gray = rotate_bound(gray, deskew_angle)
        if cfg.enable_denoise:
            gray = denoise_gray(gray, cfg)
        if cfg.enable_adaptive_threshold:
            gray = adaptive_threshold(
                gray,
                block_size=cfg.adaptive_block_size,
                c=cfg.adaptive_c,
                invert=cfg.adaptive_invert,
            )
        processed_bgr = cast(BGRImage, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    else:
        gray0 = to_gray(color_work)
        if cfg.enable_deskew:
            raw_angle = estimate_skew_deg(gray0)
            clipped = float(np.clip(raw_angle, -cfg.deskew_max_deg, cfg.deskew_max_deg))
            if abs(clipped) >= 0.05:
                deskew_angle = -clipped
                color_work = rotate_bound(color_work, deskew_angle)
        if cfg.enable_denoise:
            d = cfg.bilateral_d if cfg.bilateral_d % 2 == 1 else cfg.bilateral_d + 1
            processed_bgr = cast(
                BGRImage,
                cv2.bilateralFilter(
                    color_work,
                    d,
                    cfg.bilateral_sigma_color,
                    cfg.bilateral_sigma_space,
                ),
            )
        else:
            processed_bgr = color_work.copy()

    processed_bgr = downscale_long_edge(processed_bgr, cfg.max_long_edge_output)

    h, w = processed_bgr.shape[:2]
    do_tile = should_tile(
        h,
        w,
        tile_size=cfg.tile_size,
        min_side_threshold=cfg.tile_if_min_side_gt,
        enable=cfg.enable_tiling,
    )
    if do_tile:
        tiles = build_tiles(
            processed_bgr,
            tile_size=cfg.tile_size,
            overlap_ratio=cfg.tile_overlap_ratio,
        )
    else:
        tiles = [
            PreprocessedTile(
                row=0,
                col=0,
                offset_x=0,
                offset_y=0,
                width=w,
                height=h,
                processed_bgr=processed_bgr,
            )
        ]

    artifacts: PreprocessArtifacts | None = None
    if output_dir is not None and cfg.write_intermediates:
        stem = f"{base_name}_p{page_number:03d}"
        orig_p = output_dir / f"{stem}_original.png"
        proc_p = output_dir / f"{stem}_processed.png"
        _write_png(orig_p, original_bgr)
        _write_png(proc_p, processed_bgr)
        artifacts = PreprocessArtifacts(
            original_png=orig_p,
            processed_png=proc_p,
            source_uri=source_path,
        )

    return PreprocessPageResult(
        page_number=page_number,
        source_path=source_path,
        width=w,
        height=h,
        original_bgr=original_bgr,
        processed_bgr=processed_bgr,
        deskew_angle_deg=float(deskew_angle),
        tiles=tiles,
        artifacts=artifacts,
    )


class DrawingPreprocessPipeline:
    """Configurable preprocessing for OCR on engineering drawings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def config(self) -> PreprocessConfig:
        return self._settings.preprocess

    def process_path(
        self,
        path: Path | str,
        *,
        output_dir: Path | None = None,
        base_name: str | None = None,
    ) -> list[PreprocessPageResult]:
        """Load PDF or image, return one result per page."""
        path = Path(path)
        cfg = self.config
        name = base_name if base_name else path.stem

        if is_pdf_path(path):
            pages = pdf_to_bgr_pages(
                path,
                render_dpi=cfg.pdf_render_dpi,
                max_raster_dimension=cfg.max_raster_dimension,
            )
            return [
                _process_single_page(
                    bgr,
                    source_path=str(path.resolve()),
                    page_number=i + 1,
                    cfg=cfg,
                    output_dir=output_dir,
                    base_name=name,
                )
                for i, bgr in enumerate(pages)
            ]

        bgr = load_image_bgr(path)
        bgr = downscale_long_edge(bgr, cfg.max_raster_dimension)
        return [
            _process_single_page(
                bgr,
                source_path=str(path.resolve()),
                page_number=1,
                cfg=cfg,
                output_dir=output_dir,
                base_name=name,
            )
        ]

    def process_image_bgr(
        self,
        image_bgr: BGRImage,
        *,
        source_uri: str = "memory",
        page_number: int = 1,
        output_dir: Path | None = None,
        base_name: str = "memory",
    ) -> PreprocessPageResult:
        """Process an in-memory BGR image (tests / embedded rasters)."""
        return _process_single_page(
            image_bgr.copy(),
            source_path=source_uri,
            page_number=page_number,
            cfg=self.config,
            output_dir=output_dir,
            base_name=base_name,
        )


# Backward-compatible alias
PreprocessingPipeline = DrawingPreprocessPipeline
