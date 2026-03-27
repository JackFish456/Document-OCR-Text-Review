"""Vector storage for hybrid field matching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.vector_store.base import VectorCandidate

if TYPE_CHECKING:
    from app.vector_store.qdrant_store import QdrantStore

__all__ = ["QdrantStore", "VectorCandidate"]


def __getattr__(name: str) -> object:
    if name == "QdrantStore":
        from app.vector_store.qdrant_store import QdrantStore

        return QdrantStore
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
