"""Group extracted fields by proximity or box overlap (for layout / region workflows)."""

from __future__ import annotations

from app.models.extraction import ExtractedField
from app.spatial.fields import distance_between_fields, overlap_iou_fields


def _uf_find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: list[int], i: int, j: int) -> None:
    ri, rj = _uf_find(parent, i), _uf_find(parent, j)
    if ri != rj:
        parent[ri] = rj


def cluster_by_center_distance(
    fields: list[ExtractedField],
    *,
    max_distance_px: float,
    same_page_only: bool = True,
) -> list[list[ExtractedField]]:
    """Group fields whose **center** distance is ≤ ``max_distance_px`` (same-page pairs only).

    Uses transitive closure: if A is near B and B is near C, all three share one cluster.
    """
    if not fields:
        return []
    n = len(fields)
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if same_page_only and fields[i].page_number != fields[j].page_number:
                continue
            if distance_between_fields(fields[i], fields[j]) <= max_distance_px:
                _uf_union(parent, i, j)

    buckets: dict[int, list[ExtractedField]] = {}
    for i in range(n):
        r = _uf_find(parent, i)
        buckets.setdefault(r, []).append(fields[i])
    return list(buckets.values())


def cluster_by_overlap(
    fields: list[ExtractedField],
    *,
    min_iou: float = 0.01,
    same_page_only: bool = True,
) -> list[list[ExtractedField]]:
    """Group fields that pairwise overlap with IoU ≥ ``min_iou`` (transitive closure).

    Typical use: merge overlapping region proposals; ``min_iou`` can be ``0`` to mean any
    positive intersection area (use :meth:`~app.models.ocr.BoundingBox.intersects`).
    """
    if not fields:
        return []
    n = len(fields)
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if same_page_only and fields[i].page_number != fields[j].page_number:
                continue
            iou = overlap_iou_fields(fields[i], fields[j])
            if iou >= min_iou:
                _uf_union(parent, i, j)

    buckets: dict[int, list[ExtractedField]] = {}
    for i in range(n):
        r = _uf_find(parent, i)
        buckets.setdefault(r, []).append(fields[i])
    return list(buckets.values())
