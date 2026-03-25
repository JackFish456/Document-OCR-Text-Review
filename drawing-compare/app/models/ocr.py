"""OCR geometry and hierarchical OCR output models."""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, Field, model_validator


def _normalize_ocr_text(text: str) -> str:
    s = text.strip().lower()
    return re.sub(r"\s+", " ", s)


def normalize_ocr_text(text: str) -> str:
    """Public normalize helper for extraction, tests, and serialization pipelines."""
    return _normalize_ocr_text(text)


class BoundingBox(BaseModel):
    """Axis-aligned rectangle in pixel coordinates (inclusive-friendly corners)."""

    x1: float = Field(description="Left edge")
    y1: float = Field(description="Top edge")
    x2: float = Field(description="Right edge")
    y2: float = Field(description="Bottom edge")

    @model_validator(mode="after")
    def _valid_extent(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("BoundingBox requires x2 > x1 and y2 > y1")
        return self

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return max(0.0, self.width() * self.height())

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> BoundingBox:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        return cls(x1=x, y1=y, x2=x + width, y2=y + height)

    def to_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.width(), self.height()

    def intersects(self, other: BoundingBox) -> bool:
        return not (
            self.x2 <= other.x1
            or other.x2 <= self.x1
            or self.y2 <= other.y1
            or other.y2 <= self.y1
        )

    def center(self) -> tuple[float, float]:
        """Center point ``(cx, cy)`` in the same coordinate space as the box corners."""
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def intersection(self, other: BoundingBox) -> BoundingBox | None:
        """Axis-aligned intersection, or ``None`` if boxes are disjoint."""
        if not self.intersects(other):
            return None
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def intersection_area(self, other: BoundingBox) -> float:
        """Area of overlap (0 if disjoint)."""
        inter = self.intersection(other)
        return inter.area() if inter is not None else 0.0

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-union in ``[0, 1]``."""
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union_area = self.area() + other.area() - inter
        return inter / union_area if union_area > 0.0 else 0.0

    def union(self, other: BoundingBox) -> BoundingBox:
        """Smallest axis-aligned box containing both inputs."""
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )


class OCRToken(BaseModel):
    """Smallest OCR unit (word / token)."""

    text: str = Field(min_length=1)
    normalized_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    line_id: str = ""
    block_id: str = ""

    @model_validator(mode="after")
    def _fill_normalized(self) -> Self:
        if not self.normalized_text:
            self.normalized_text = _normalize_ocr_text(self.text)
        return self


class OCRLine(BaseModel):
    """Logical text line composed of tokens."""

    id: str = Field(min_length=1)
    text: str = ""
    normalized_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: BoundingBox
    tokens: list[OCRToken] = Field(default_factory=list)

    @model_validator(mode="after")
    def _aggregate_from_tokens(self) -> Self:
        if self.tokens:
            if not self.text:
                self.text = " ".join(t.text for t in self.tokens)
            if not self.normalized_text:
                self.normalized_text = _normalize_ocr_text(self.text)
            confs = [t.confidence for t in self.tokens if t.confidence is not None]
            if confs and self.confidence == 0.0:
                self.confidence = sum(confs) / len(confs)
        elif not self.normalized_text:
            self.normalized_text = _normalize_ocr_text(self.text)
        return self

    def with_token_line_ids(self) -> OCRLine:
        """Return a copy of the line with token line_id aligned to this line."""
        tok = [t.model_copy(update={"line_id": self.id}) for t in self.tokens]
        return self.model_copy(update={"tokens": tok})


class OCRPage(BaseModel):
    """Single raster page or vector sheet render."""

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    lines: list[OCRLine] = Field(default_factory=list)
    tokens: list[OCRToken] = Field(default_factory=list)

    @model_validator(mode="after")
    def _tokens_consistent(self) -> Self:
        from_line = [t for line in self.lines for t in line.tokens]
        if not self.lines:
            return self
        if not self.tokens:
            self.tokens = from_line
            return self
        if len(self.tokens) != len(from_line):
            raise ValueError("OCRPage.tokens length must match flattened line.tokens")
        return self


class OCRDocument(BaseModel):
    """Full OCR output for one source file."""

    document_id: str = Field(min_length=1)
    source_path: str = ""
    provider_name: str = ""
    pages: list[OCRPage] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    def all_tokens_flat(self) -> list[OCRToken]:
        return [t for p in self.pages for t in p.tokens]

    def all_lines_flat(self) -> list[tuple[int, OCRLine]]:
        return [(p.page_number, line) for p in self.pages for line in p.lines]


class OcrRegion(BaseModel):
    """Lightweight region for OCR providers that do not emit full document structure."""

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    page_number: int = Field(default=1, ge=1)

    def to_token(self) -> OCRToken:
        return OCRToken(
            text=self.text,
            normalized_text=_normalize_ocr_text(self.text),
            confidence=self.confidence,
            bbox=self.bbox,
        )
