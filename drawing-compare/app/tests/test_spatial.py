"""BoundingBox geometry, spatial utilities, and optional field spatial metadata."""

from __future__ import annotations

import json

import pytest

from app.models.extraction import ExtractedField
from app.models.ocr import BoundingBox
from app.models.serialization import model_from_json, model_to_json
from app.models.spatial_metadata import SpatialMetadata
from app.spatial import (
    cluster_by_center_distance,
    cluster_by_overlap,
    distance_between_fields,
    overlap_iou_fields,
)


def test_bounding_box_center_intersection_iou_union() -> None:
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    assert a.center() == (5.0, 5.0)
    b = BoundingBox(x1=5, y1=5, x2=15, y2=15)
    inter = a.intersection(b)
    assert inter is not None
    assert inter.area() == 25.0
    assert a.iou(b) == pytest.approx(25.0 / (100 + 100 - 25))
    u = a.union(b)
    assert u.x1 == 0 and u.y1 == 0 and u.x2 == 15 and u.y2 == 15


def test_bounding_box_disjoint_iou_zero() -> None:
    a = BoundingBox(x1=0, y1=0, x2=1, y2=1)
    b = BoundingBox(x1=5, y1=5, x2=6, y2=6)
    assert a.intersection(b) is None
    assert a.iou(b) == 0.0


def test_distance_and_overlap_fields() -> None:
    fa = ExtractedField(
        field_id="a",
        value="x",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=1.0,
        page_number=1,
    )
    fb = ExtractedField(
        field_id="b",
        value="y",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=1.0,
        page_number=1,
    )
    assert distance_between_fields(fa, fb) == 0.0
    assert overlap_iou_fields(fa, fb) == 1.0

    fc = ExtractedField(
        field_id="c",
        value="z",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=1.0,
        page_number=2,
    )
    assert distance_between_fields(fa, fc) == float("inf")
    assert overlap_iou_fields(fa, fc) == 0.0


def test_cluster_by_center_distance() -> None:
    f0 = ExtractedField(
        field_id="0",
        value="a",
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        confidence=1.0,
        page_number=1,
    )
    f1 = ExtractedField(
        field_id="1",
        value="b",
        bbox=BoundingBox(x1=2, y1=0, x2=3, y2=1),
        confidence=1.0,
        page_number=1,
    )
    f2 = ExtractedField(
        field_id="2",
        value="c",
        bbox=BoundingBox(x1=100, y1=0, x2=101, y2=1),
        confidence=1.0,
        page_number=1,
    )
    clusters = cluster_by_center_distance([f0, f1, f2], max_distance_px=3.0)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_overlap() -> None:
    f0 = ExtractedField(
        field_id="0",
        value="a",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=1.0,
        page_number=1,
    )
    f1 = ExtractedField(
        field_id="1",
        value="b",
        bbox=BoundingBox(x1=5, y1=5, x2=15, y2=15),
        confidence=1.0,
        page_number=1,
    )
    clusters = cluster_by_overlap([f0, f1], min_iou=0.01)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_extracted_field_spatial_round_trip() -> None:
    meta = SpatialMetadata(cluster_id=3, layout_zone="title_block", attributes={"layer": "diff"})
    f = ExtractedField(
        field_id="x",
        value="v",
        bbox=BoundingBox(x1=0, y1=0, x2=4, y2=4),
        confidence=1.0,
        spatial=meta,
    )
    raw = model_to_json(f, indent=None)
    g = model_from_json(ExtractedField, raw)
    assert g.spatial is not None
    assert g.spatial.cluster_id == 3
    assert g.spatial.layout_zone == "title_block"
    assert g.bbox.x2 == 4


def test_extracted_field_default_no_spatial_json() -> None:
    f = ExtractedField(
        field_id="x",
        value="v",
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        confidence=1.0,
    )
    raw = model_to_json(f, indent=None)
    assert "spatial" not in json.loads(raw)
