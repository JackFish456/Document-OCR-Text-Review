"""Classify aligned pairs into comparison labels."""

from app.classification.discrepancy_classifier import (
    DiscrepancyClassificationConfig,
    DiscrepancyClassifier,
    discrepancy_flags_for_matches,
)
from app.classification.field_classifier import classify_pair

__all__ = [
    "DiscrepancyClassificationConfig",
    "DiscrepancyClassifier",
    "classify_pair",
    "discrepancy_flags_for_matches",
]
