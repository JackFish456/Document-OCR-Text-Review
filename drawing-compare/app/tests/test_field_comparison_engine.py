"""Unit tests for :mod:`app.matching.field_comparison_engine`."""

from __future__ import annotations

from app.matching.field_comparison_engine import (
    FieldComparisonConfig,
    FieldComparisonEngine,
    compare_field_lists,
    compute_pair_breakdown,
    regex_compatibility,
)
from app.models.extraction import ExtractedField
from app.models.match import MatchType
from app.models.ocr import BoundingBox


def _bbox() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=10, y2=10)


def test_regex_compatibility_same_pattern() -> None:
    assert regex_compatibility("12 x 24", "6 x 8") == 1.0
    assert regex_compatibility("2024-01-02", "2025-03-01") == 1.0


def test_regex_compatibility_mixed() -> None:
    assert regex_compatibility("12 x 24", "plain text") == 0.5


def test_compute_pair_breakdown_composite_weights() -> None:
    sa = ExtractedField(
        field_id="a",
        label="Drawing No",
        value="A-100",
        normalized_label="drawing no",
        normalized_value="a-100",
        bbox=_bbox(),
        confidence=1.0,
    )
    sb = ExtractedField(
        field_id="b",
        label="Drawing No",
        value="A-100",
        normalized_label="drawing no",
        normalized_value="a-100",
        bbox=_bbox(),
        confidence=1.0,
    )
    cfg = FieldComparisonConfig()
    bd = compute_pair_breakdown(sa, sb, cfg)
    assert bd.label_similarity >= 0.99
    assert bd.value_similarity >= 0.99
    assert bd.composite_score >= 0.95


def test_exact_match_value_only() -> None:
    fa = ExtractedField(
        field_id="a",
        value="Project 001",
        raw_text="Project 001",
        normalized_value="project 001",
        bbox=_bbox(),
        confidence=1.0,
    )
    fb = ExtractedField(
        field_id="b",
        value="Project 001",
        raw_text="Project 001",
        normalized_value="project 001",
        bbox=_bbox(),
        confidence=1.0,
    )
    engine = FieldComparisonEngine()
    results = engine.build_match_results([fa], [fb])
    assert len(results) == 1
    assert results[0].match_type == MatchType.EXACT_MATCH
    assert results[0].reason and "classification=exact_match" in results[0].reason


def test_changed_value_same_label() -> None:
    fa = ExtractedField(
        field_id="a",
        label="REVISION",
        value="A",
        normalized_label="revision",
        normalized_value="a",
        bbox=_bbox(),
        confidence=1.0,
    )
    fb = ExtractedField(
        field_id="b",
        label="REVISION",
        value="C",
        normalized_label="revision",
        normalized_value="c",
        bbox=_bbox(),
        confidence=1.0,
    )
    engine = FieldComparisonEngine()
    results = engine.build_match_results([fa], [fb])
    assert results[0].match_type == MatchType.CHANGED_VALUE
    assert "normalized_label_similarity" in (results[0].reason or "")


def test_missing_and_extra() -> None:
    only_src = ExtractedField(
        field_id="s",
        value="Only source",
        normalized_value="only source",
        bbox=_bbox(),
        confidence=1.0,
    )
    only_tgt = ExtractedField(
        field_id="t",
        value="Only target",
        normalized_value="only target",
        bbox=_bbox(),
        confidence=1.0,
    )
    engine = FieldComparisonEngine(FieldComparisonConfig(min_pair_composite=0.9))
    results = engine.build_match_results([only_src], [only_tgt])
    types = {r.match_type for r in results}
    assert MatchType.MISSING_IN_TARGET in types
    assert MatchType.EXTRA_IN_TARGET in types


def test_compare_field_lists_builds_report() -> None:
    fa = ExtractedField(
        field_id="a",
        value="X",
        normalized_value="x",
        bbox=_bbox(),
        confidence=1.0,
    )
    fb = ExtractedField(
        field_id="b",
        value="X",
        normalized_value="x",
        bbox=_bbox(),
        confidence=1.0,
    )
    report = compare_field_lists([fa], [fb])
    assert report.summary.total_source == 1
    assert report.summary.matched >= 1


def test_partial_match_typo_in_label() -> None:
    fa = ExtractedField(
        field_id="a",
        label="Sheet Title",
        value="North Elevation",
        normalized_label="sheet title",
        normalized_value="north elevation",
        bbox=_bbox(),
        confidence=1.0,
    )
    fb = ExtractedField(
        field_id="b",
        label="Sheet Titl",
        value="North Elevation",
        normalized_label="sheet titl",
        normalized_value="north elevation",
        bbox=_bbox(),
        confidence=1.0,
    )
    engine = FieldComparisonEngine()
    results = engine.build_match_results([fa], [fb])
    assert results[0].match_type in (MatchType.PARTIAL_MATCH, MatchType.EXACT_MATCH)


def test_uncertain_low_composite_config() -> None:
    fa = ExtractedField(
        field_id="a",
        label="A",
        value="1",
        normalized_label="a",
        normalized_value="1",
        bbox=_bbox(),
        confidence=1.0,
    )
    fb = ExtractedField(
        field_id="b",
        label="Z",
        value="999",
        normalized_label="z",
        normalized_value="999",
        bbox=_bbox(),
        confidence=1.0,
    )
    cfg = FieldComparisonConfig(
        min_pair_composite=0.01,
        partial_composite_min=0.99,
        uncertain_composite_max=0.99,
    )
    engine = FieldComparisonEngine(cfg)
    results = engine.build_match_results([fa], [fb])
    assert results[0].match_type == MatchType.UNCERTAIN
