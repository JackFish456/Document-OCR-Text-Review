"""Spatial helpers for geometry and future layout-aware comparison (boxes preserved on models)."""

from app.spatial.clustering import cluster_by_center_distance, cluster_by_overlap
from app.spatial.fields import (
    boxes_overlap,
    center_point,
    distance_between_boxes,
    distance_between_fields,
    overlap_iou,
    overlap_iou_fields,
)

__all__ = [
    "boxes_overlap",
    "center_point",
    "cluster_by_center_distance",
    "cluster_by_overlap",
    "distance_between_boxes",
    "distance_between_fields",
    "overlap_iou",
    "overlap_iou_fields",
]
