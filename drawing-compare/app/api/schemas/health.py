"""Typed health check response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service liveness payload."""

    status: Literal["ok"] = "ok"
    service: str = Field(default="drawing-compare", description="Logical service name")
    version: str = Field(description="Application version string")
