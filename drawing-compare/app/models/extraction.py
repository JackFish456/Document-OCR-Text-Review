"""Structured fields extracted from OCR (title block, notes, tags, etc.)."""

from __future__ import annotations

from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.ocr import BoundingBox, OCRToken, _normalize_ocr_text
from app.models.spatial_metadata import SpatialMetadata
from app.text_normalization import normalize_label, normalize_value


class ExtractedField(BaseModel):
    """A single semantic field after extraction / parsing."""

    field_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    field_type: str = Field(
        default="generic",
        description="Domain field class, e.g. title_block, revision, scale, note",
    )
    label: str = ""
    value: str = ""
    raw_text: str = ""
    normalized_label: str = ""
    normalized_value: str = ""
    page_number: int = Field(default=1, ge=1)
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_rule: str = Field(
        default="",
        description="Identifier of the line extractor rule that produced this field",
    )
    spatial: SpatialMetadata | None = Field(
        default=None,
        description="Optional layout/cluster/overlay hints; bbox remains the geometry of record",
    )

    @model_validator(mode="after")
    def _defaults_from_raw(self) -> Self:
        if not self.raw_text and (self.label or self.value):
            raw = f"{self.label}: {self.value}".strip().strip(":")
            self.raw_text = raw
        if not self.value and self.raw_text and not self.label:
            self.value = self.raw_text
        if not self.normalized_value and self.value:
            self.normalized_value = normalize_value(self.value).normalized
        if not self.normalized_label and self.label:
            self.normalized_label = normalize_label(self.label).normalized
        return self

    @field_validator("normalized_value", "normalized_label", mode="before")
    @classmethod
    def _strip_norm(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    def bbox_center(self) -> tuple[float, float]:
        """Center ``(cx, cy)`` of :attr:`bbox` (same coordinates as stored corners)."""
        return self.bbox.center()

    def comparison_key(self) -> str:
        """Primary string used for fuzzy alignment (value-first, then label)."""
        if self.label and self.value:
            ln = (self.normalized_label or normalize_label(self.label).normalized).replace("|", "/")
            vn = (self.normalized_value or normalize_value(self.value).normalized).replace("|", "/")
            return f"{ln}|{vn}"
        if self.normalized_value:
            return self.normalized_value
        if self.normalized_label:
            return self.normalized_label
        return _normalize_ocr_text(self.raw_text or self.value or self.label)

    @classmethod
    def from_ocr_token(
        cls,
        token: OCRToken,
        *,
        field_type: str = "ocr_token",
        page_number: int = 1,
    ) -> ExtractedField:
        return cls(
            field_id=str(uuid4()),
            field_type=field_type,
            label="",
            value=token.text,
            raw_text=token.text,
            normalized_value=token.normalized_text,
            page_number=page_number,
            bbox=token.bbox,
            confidence=token.confidence,
        )
