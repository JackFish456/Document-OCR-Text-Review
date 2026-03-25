"""Optional spatial hints attached to extracted fields (layout / clustering / overlays)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SpatialMetadata(BaseModel):
    """Non-authoritative metadata for location-aware workflows.

    **Does not replace** :attr:`~app.models.extraction.ExtractedField.bbox`; geometry of record
    remains on :class:`~app.models.ocr.BoundingBox`. This block holds optional labels for future
    layout comparison, region matching, and visual diff layers.
    """

    coordinate_space: str = Field(
        default="page_pixel",
        description="Interpretation of bbox coordinates, e.g. page_pixel | normalized_01",
    )
    cluster_id: int | None = Field(default=None, description="Optional cluster from spatial grouping")
    layout_zone: str | None = Field(
        default=None,
        description="Optional semantic region, e.g. title_block, notes_column",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Forward-compatible key/value bag for overlay styles, layer ids, etc.",
    )
