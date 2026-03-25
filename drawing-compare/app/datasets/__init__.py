"""Golden dataset manifests, loaders, and validation."""

from app.datasets.manifest import GoldenDatasetLoader, ResolvedGoldenPair
from app.datasets.validator import (
    GoldenValidationResult,
    validate_annotation,
    validate_annotation_matches_manifest,
    validate_golden_dataset,
    validate_index,
    validate_manifest_entries,
    validate_manifest_file,
)

__all__ = [
    "GoldenDatasetLoader",
    "GoldenValidationResult",
    "ResolvedGoldenPair",
    "validate_annotation",
    "validate_annotation_matches_manifest",
    "validate_golden_dataset",
    "validate_index",
    "validate_manifest_entries",
    "validate_manifest_file",
]
