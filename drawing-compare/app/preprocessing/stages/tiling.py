"""Overlap tiling for very large processed sheets."""

from __future__ import annotations

from app.preprocessing.types import BGRImage, PreprocessedTile


def build_tiles(
    processed_bgr: BGRImage,
    *,
    tile_size: int,
    overlap_ratio: float,
) -> list[PreprocessedTile]:
    """Split processed BGR into overlapping tiles (page coordinates)."""
    h, w = processed_bgr.shape[:2]
    if h <= tile_size and w <= tile_size:
        return [
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

    step = max(1, int(round(tile_size * (1.0 - overlap_ratio))))
    tiles: list[PreprocessedTile] = []
    row_idx = 0
    for y in range(0, h, step):
        col_idx = 0
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            if y_end <= y or x_end <= x:
                continue
            crop = processed_bgr[y:y_end, x:x_end].copy()
            tiles.append(
                PreprocessedTile(
                    row=row_idx,
                    col=col_idx,
                    offset_x=x,
                    offset_y=y,
                    width=crop.shape[1],
                    height=crop.shape[0],
                    processed_bgr=crop,
                )
            )
            col_idx += 1
            if x_end >= w:
                break
        row_idx += 1
        if y + tile_size >= h:
            break

    if not tiles:
        return [
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
    return tiles


def should_tile(
    height: int,
    width: int,
    *,
    tile_size: int,
    min_side_threshold: int,
    enable: bool,
) -> bool:
    if not enable:
        return False
    return min(height, width) >= min_side_threshold or max(height, width) > tile_size
