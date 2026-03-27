"""Field text embedding for hybrid / vector retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.embeddings.text_embedder import TextEmbedder

__all__ = ["TextEmbedder", "clear_text_embedder_cache", "get_text_embedder"]


def __getattr__(name: str) -> object:
    if name == "TextEmbedder":
        from app.embeddings.text_embedder import TextEmbedder

        return TextEmbedder
    if name in ("clear_text_embedder_cache", "get_text_embedder"):
        from app.embeddings import factory as _factory

        return getattr(_factory, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
