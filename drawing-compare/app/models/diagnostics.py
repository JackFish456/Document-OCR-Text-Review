"""Structured outputs from the compare pipeline for evaluation and error analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.extraction import ExtractedField


class PipelineDiagnostics(BaseModel):
    """Fields produced by parsing after OCR — used to separate extraction vs matching failures."""

    source_fields: list[ExtractedField] = Field(default_factory=list)
    target_fields: list[ExtractedField] = Field(default_factory=list)
