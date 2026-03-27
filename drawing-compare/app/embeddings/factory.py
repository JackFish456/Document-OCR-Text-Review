"""Config-backed :class:`~app.embeddings.text_embedder.TextEmbedder` factory."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import get_settings

if TYPE_CHECKING:
    from app.embeddings.text_embedder import TextEmbedder


@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    """Singleton embedder for the current :func:`~app.core.config.get_settings` model name."""
    from app.embeddings.text_embedder import TextEmbedder

    return TextEmbedder(model_name=get_settings().EMBEDDING_MODEL)


def clear_text_embedder_cache() -> None:
    """Clear cached embedder (e.g. after tests change ``EMBEDDING_MODEL``)."""
    get_text_embedder.cache_clear()
