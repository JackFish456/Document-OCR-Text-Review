"""Shared types for vector retrieval."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VectorCandidate(BaseModel):
    """One hit from vector similarity search."""

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)
