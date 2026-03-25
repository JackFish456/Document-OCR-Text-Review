"""Preprocessing pipeline unit tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.preprocessing.deskew import estimate_skew_deg
from app.preprocessing.io import load_image_bgr
from app.preprocessing.ocr_run import run_ocr_on_preprocessed_page, shift_ocr_regions
from app.preprocessing.pipeline import DrawingPreprocessPipeline
from app.preprocessing.settings import PreprocessConfig
from app.preprocessing.stages.grayscale import to_gray
from app.preprocessing.stages.threshold import adaptive_threshold
from app.preprocessing.stages.tiling import build_tiles, should_tile
from app.ocr.stub_provider import EchoOcrProvider
from app.models.ocr import BoundingBox, OcrRegion


def test_adaptive_threshold_output_shape() -> None:
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    cv2.putText(img, "A102", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 2)
    gray = to_gray(img)
    bw = adaptive_threshold(gray, block_size=35, c=10, invert=False)
    assert bw.shape == gray.shape
    assert bw.dtype == np.uint8


def test_build_tiles_single_when_small() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    tiles = build_tiles(img, tile_size=256, overlap_ratio=0.1)
    assert len(tiles) == 1
    assert tiles[0].offset_x == 0


def test_should_tile_respects_flag() -> None:
    assert should_tile(5000, 5000, tile_size=1024, min_side_threshold=4096, enable=False) is False
    assert should_tile(5000, 5000, tile_size=1024, min_side_threshold=4096, enable=True) is True


def test_shift_ocr_regions() -> None:
    regs = [
        OcrRegion(
            text="x",
            confidence=1.0,
            bbox=BoundingBox(x1=1, y1=2, x2=5, y2=6),
        )
    ]
    out = shift_ocr_regions(regs, 10, 20)
    assert out[0].bbox.x1 == 11


def test_pipeline_process_image_in_memory(tmp_path: Path) -> None:
    img = np.zeros((200, 240, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    cv2.rectangle(img, (30, 30), (210, 120), (0, 0, 0), 2)
    cfg = PreprocessConfig(
        enable_deskew=False,
        enable_denoise=False,
        max_long_edge_output=1024,
        write_intermediates=True,
    )
    settings = Settings(preprocess=cfg)
    pipe = DrawingPreprocessPipeline(settings)
    out = pipe.process_image_bgr(
        img,
        source_uri="test",
        output_dir=tmp_path,
        base_name="unit",
    )
    assert out.height == 200
    assert len(out.tiles) >= 1
    assert out.artifacts is not None
    assert out.artifacts.processed_png is not None
    assert out.artifacts.processed_png.exists()


def test_run_ocr_merges_tiles() -> None:
    cfg = PreprocessConfig(
        enable_grayscale=False,
        enable_clahe=False,
        enable_deskew=False,
        enable_denoise=False,
        enable_adaptive_threshold=False,
        enable_tiling=True,
        tile_size=64,
        tile_overlap_ratio=0.1,
        tile_if_min_side_gt=50,
        max_long_edge_output=0,
    )
    settings = Settings(preprocess=cfg)
    pipe = DrawingPreprocessPipeline(settings)
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)
    page = pipe.process_image_bgr(img, source_uri="tiled")
    assert len(page.tiles) >= 2
    ocr = EchoOcrProvider()
    regions = run_ocr_on_preprocessed_page(ocr, page)
    assert len(regions) >= 1


@pytest.mark.skipif(not Path("data/samples").exists(), reason="sample file optional")
def test_load_sample_image_if_present() -> None:
    samples = Path("data/samples")
    for p in samples.glob("*.png"):
        bgr = load_image_bgr(p)
        assert bgr.ndim == 3
        break
    else:
        pytest.skip("no png in data/samples")


def test_estimate_skew_near_zero_on_axis_aligned() -> None:
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (40, 50), (260, 80), 255, 3)
    assert abs(estimate_skew_deg(img)) < 20.0
