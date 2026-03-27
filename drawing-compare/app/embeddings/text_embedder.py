"""Sentence-transformer embeddings for field text."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.extraction import ExtractedField


class TextEmbedder:
    """Loads a config-driven model and embeds raw text or canonical field strings."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def canonical_field_text(field: ExtractedField) -> str:
        """Build normalized ``type=… label=… value=…`` string for embedding."""
        t = field.field_type.strip().lower()
        label = field.label.strip().lower()
        value = field.value.strip().lower()
        return f"type={t} label={label} value={value}"

    def embed(self, text: str) -> list[float]:
        normalized = self._normalize_text(text)
        vec = self._model.encode(normalized, convert_to_numpy=True)
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        return arr.tolist()

    def embed_field(self, field: ExtractedField) -> list[float]:
        return self.embed(self.canonical_field_text(field))
