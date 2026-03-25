"""Liveness and readiness style health checks."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return process liveness and build version (for load balancers and operators)."""
    return HealthResponse(version=__version__)
