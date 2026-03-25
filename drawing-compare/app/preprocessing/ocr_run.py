"""Run OCR providers on preprocessed pages (full frame or tiles with offset merge)."""

from __future__ import annotations

from app.models.ocr import BoundingBox, OcrRegion
from app.ocr.base import OcrProvider
from app.preprocessing.types import PreprocessPageResult


def shift_ocr_regions(regions: list[OcrRegion], dx: float, dy: float) -> list[OcrRegion]:
    """Translate region boxes into page coordinates."""
    out: list[OcrRegion] = []
    for r in regions:
        b = r.bbox
        out.append(
            r.model_copy(
                update={
                    "bbox": BoundingBox(
                        x1=b.x1 + dx,
                        y1=b.y1 + dy,
                        x2=b.x2 + dx,
                        y2=b.y2 + dy,
                    )
                }
            )
        )
    return out


def run_ocr_on_preprocessed_page(ocr: OcrProvider, page: PreprocessPageResult) -> list[OcrRegion]:
    """Invoke OCR on each tile (or full-page singleton) and merge boxes into page space."""
    if len(page.tiles) == 1 and page.tiles[0].offset_x == 0 and page.tiles[0].offset_y == 0:
        return ocr.run(page.tiles[0].processed_bgr)

    regions: list[OcrRegion] = []
    for tile in page.tiles:
        chunk = ocr.run(tile.processed_bgr)
        regions.extend(shift_ocr_regions(chunk, float(tile.offset_x), float(tile.offset_y)))
    return regions
