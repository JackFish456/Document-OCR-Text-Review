"""Matching and classification unit tests."""

from app.matching.align import align_fields
from app.matching.field_comparison_engine import FieldComparisonConfig, FieldComparisonEngine
from app.models.extraction import ExtractedField
from app.models.match import MatchType
from app.models.ocr import BoundingBox


def test_align_and_classify_exact() -> None:
    fa = ExtractedField(
        field_id="a",
        field_type="generic",
        value="Project 001",
        raw_text="Project 001",
        normalized_value="project 001",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=1.0,
    )
    fb = ExtractedField(
        field_id="b",
        field_type="generic",
        value="Project 001",
        raw_text="Project 001",
        normalized_value="project 001",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=1.0,
    )
    aln = align_fields([fa], [fb])
    assert len(aln.pairs) == 1
    engine = FieldComparisonEngine()
    result = engine.build_match_results([fa], [fb])[0]
    assert result.match_type == MatchType.EXACT_MATCH


def test_classify_missing() -> None:
    fa = ExtractedField(
        field_id="a",
        field_type="generic",
        value="Only A",
        raw_text="Only A",
        normalized_value="only a",
        bbox=BoundingBox(x1=0, y1=0, x2=5, y2=5),
        confidence=1.0,
    )
    engine = FieldComparisonEngine(FieldComparisonConfig(min_pair_composite=0.99))
    result = engine.build_match_results([fa], [])[0]
    assert result.match_type == MatchType.MISSING_IN_TARGET
