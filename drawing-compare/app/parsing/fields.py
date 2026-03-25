"""Heuristic parser from OCR regions to `ExtractedField` list."""

from __future__ import annotations

from uuid import uuid4

from app.models.extraction import ExtractedField
from app.models.ocr import OcrRegion
from app.text_normalization import normalize_value


class ParsedDrawing:
    """Container for parsed fields from one sheet."""

    def __init__(self, fields: list[ExtractedField]) -> None:
        self.fields = fields


class RegionParser:
    """Baseline parser: one field per OCR region (extend with title-block logic)."""

    def parse(self, regions: list[OcrRegion]) -> ParsedDrawing:
        fields: list[ExtractedField] = []
        for i, region in enumerate(regions):
            fid = f"auto-{i}-{uuid4().hex[:8]}"
            raw = region.text
            fields.append(
                ExtractedField(
                    field_id=fid,
                    field_type="ocr_region",
                    label="",
                    value=raw,
                    raw_text=raw,
                    normalized_value=normalize_value(raw).normalized,
                    page_number=region.page_number,
                    bbox=region.bbox,
                    confidence=region.confidence,
                )
            )
        return ParsedDrawing(fields=fields)
