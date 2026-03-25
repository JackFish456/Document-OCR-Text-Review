"""Pydantic domain models."""

from app.models.comparison import CompareRequest, CompareResponse
from app.models.evaluation import (
    EvaluationRunSummary,
    FieldLevelMetric,
    GoldenFieldExpectation,
    GoldenManifestDocument,
    GoldenManifestIndex,
    GoldenManifestIndexEntry,
    GoldenPairAnnotation,
    GoldenPairManifestEntry,
)
from app.models.extraction import ExtractedField
from app.models.spatial_metadata import SpatialMetadata
from app.models.golden import GoldenExpectedLabel, GoldenPair, GoldenPairExpected
from app.models.match import ComparisonReport, ComparisonSummary, MatchResult, MatchType
from app.models.ocr import (
    BoundingBox,
    OCRDocument,
    OCRLine,
    OCRPage,
    OCRToken,
    OcrRegion,
    normalize_ocr_text,
)
from app.models.review_flag import ReviewFlag, ReviewFlagSeverity, ReviewFlagType
from app.models.serialization import (
    dump_many,
    load_many,
    model_from_dict,
    model_from_json,
    model_to_dict,
    model_to_json,
)

__all__ = [
    "BoundingBox",
    "CompareRequest",
    "CompareResponse",
    "ComparisonReport",
    "ComparisonSummary",
    "EvaluationRunSummary",
    "ExtractedField",
    "FieldLevelMetric",
    "GoldenExpectedLabel",
    "GoldenFieldExpectation",
    "GoldenManifestDocument",
    "GoldenManifestIndex",
    "GoldenManifestIndexEntry",
    "GoldenPair",
    "GoldenPairAnnotation",
    "GoldenPairExpected",
    "GoldenPairManifestEntry",
    "MatchResult",
    "MatchType",
    "OCRDocument",
    "OCRLine",
    "OCRPage",
    "OCRToken",
    "OcrRegion",
    "ReviewFlag",
    "ReviewFlagSeverity",
    "ReviewFlagType",
    "SpatialMetadata",
    "dump_many",
    "load_many",
    "model_from_dict",
    "model_from_json",
    "model_to_dict",
    "model_to_json",
    "normalize_ocr_text",
]
