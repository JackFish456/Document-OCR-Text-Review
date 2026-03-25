"""Field alignment and similarity scoring."""

from app.matching.align import FieldAlignment, FieldCandidatePair, align_fields
from app.matching.field_comparison_engine import (
    FieldComparisonConfig,
    FieldComparisonEngine,
    PairScoreBreakdown,
    compare_field_lists,
    compute_pair_breakdown,
    regex_compatibility,
)

__all__ = [
    "FieldAlignment",
    "FieldCandidatePair",
    "FieldComparisonConfig",
    "FieldComparisonEngine",
    "PairScoreBreakdown",
    "align_fields",
    "compare_field_lists",
    "compute_pair_breakdown",
    "regex_compatibility",
]
