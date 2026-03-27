"""OCR provider protocol and shared types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from app.models.ocr import OcrRegion

OcrImageInput = NDArray[np.uint8]

BBoxInt = tuple[int, int, int, int]


class OCRResult(BaseModel):
    """Normalized, provider-agnostic OCR output for a whole document (parallel to legacy ``OCRDocument``)."""

    class Token(BaseModel):
        text: str = Field(min_length=1)
        bbox: BBoxInt
        confidence: float = Field(ge=0.0, le=1.0)

        @field_validator("bbox")
        @classmethod
        def _bbox_ordered(cls, v: BBoxInt) -> BBoxInt:
            x1, y1, x2, y2 = v
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox requires x2 > x1 and y2 > y1")
            return v

    class Line(BaseModel):
        text: str
        bbox: BBoxInt
        confidence: float = Field(ge=0.0, le=1.0)
        tokens: list["OCRResult.Token"] = Field(default_factory=list)

        @field_validator("bbox")
        @classmethod
        def _bbox_ordered(cls, v: BBoxInt) -> BBoxInt:
            x1, y1, x2, y2 = v
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox requires x2 > x1 and y2 > y1")
            return v

    class Page(BaseModel):
        page_number: int = Field(ge=1)
        width: int = Field(gt=0)
        height: int = Field(gt=0)
        lines: list["OCRResult.Line"] = Field(default_factory=list)

    pages: list[Page] = Field(default_factory=list)


OCRResult.model_rebuild()


class OCRProvider(ABC):
    """Document-level OCR: read a file path and return a normalized :class:`OCRResult`."""

    @abstractmethod
    def extract(self, document_path: str) -> OCRResult:
        """Run OCR on ``document_path`` and return normalized pages, lines, and tokens."""
        ...


@runtime_checkable
class OcrProvider(Protocol):
    """Pluggable OCR backend (Tesseract, cloud APIs, etc.)."""

    provider_name: str

    def run(self, image: OcrImageInput, *, language_hint: str | None = None) -> list[OcrRegion]:
        """Run OCR on a single-channel or BGR image array."""
        ...
