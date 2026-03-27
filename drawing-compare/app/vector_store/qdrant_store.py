"""Qdrant-backed storage for per-field embeddings."""

from __future__ import annotations

import uuid
from uuid import NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings, get_settings
from app.models.extraction import ExtractedField
from app.vector_store.base import VectorCandidate


def _field_text(field: ExtractedField) -> str:
    raw = (field.raw_text or "").strip()
    if raw:
        return raw
    if field.label and field.value:
        return f"{field.label}: {field.value}".strip()
    return (field.value or field.label or "").strip()


def _point_id(doc_id: str, field_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE_URL, f"{doc_id}\0{field_id}"))


class QdrantStore:
    """Connects to Qdrant, ensures a collection exists, upserts fields, and searches by vector."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = QdrantClient(url=self._settings.QDRANT_URL)
        self._collection_name = self._settings.QDRANT_COLLECTION_PREFIX

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(collection_name=self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_fields(
        self,
        doc_id: str,
        fields: list[ExtractedField],
        embeddings: list[list[float]],
    ) -> None:
        if len(fields) != len(embeddings):
            msg = f"fields ({len(fields)}) and embeddings ({len(embeddings)}) length mismatch"
            raise ValueError(msg)
        if not fields:
            return
        dim = len(embeddings[0])
        if any(len(e) != dim for e in embeddings):
            msg = "All embeddings must share the same dimension"
            raise ValueError(msg)
        self._ensure_collection(dim)
        points: list[PointStruct] = []
        for field, vector in zip(fields, embeddings, strict=True):
            payload = {
                "text": _field_text(field),
                "field_type": field.field_type,
                "label": field.label,
                "value": field.value,
                "page": field.page_number,
                "doc_id": doc_id,
            }
            points.append(
                PointStruct(
                    id=_point_id(doc_id, field.field_id),
                    vector=vector,
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection_name, points=points)

    def query_similar(
        self,
        doc_id: str,
        vector: list[float],
        top_k: int,
    ) -> list[VectorCandidate]:
        """Return top similar points for ``doc_id`` (payload filter), by score descending."""
        if top_k < 1:
            msg = "top_k must be >= 1"
            raise ValueError(msg)
        if not self._client.collection_exists(collection_name=self._collection_name):
            return []
        flt = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
        )
        hits = self._client.search(
            collection_name=self._collection_name,
            query_vector=vector,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        out: list[VectorCandidate] = []
        for h in hits:
            pl = dict(h.payload) if h.payload is not None else {}
            out.append(VectorCandidate(id=str(h.id), score=float(h.score), payload=pl))
        return out
