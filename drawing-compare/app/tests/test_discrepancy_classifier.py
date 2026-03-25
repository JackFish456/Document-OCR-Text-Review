"""Tests for :mod:`app.classification.discrepancy_classifier`."""

from __future__ import annotations

from app.classification.discrepancy_classifier import (
    DiscrepancyClassificationConfig,
    DiscrepancyClassifier,
    discrepancy_flags_for_matches,
)
from app.models.extraction import ExtractedField
from app.models.match import MatchResult, MatchType
from app.models.ocr import BoundingBox
from app.models.review_flag import ReviewFlagSeverity, ReviewFlagType


def _box() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=1, y2=1)


def _pair(
    *,
    match_type: MatchType,
    confidence: float = 0.9,
    reason: str | None = "engine",
    src_type: str = "generic",
    tgt_type: str = "generic",
    src_label: str = "ITEM",
    tgt_label: str = "ITEM",
    src_value: str = "1",
    tgt_value: str = "2",
) -> MatchResult:
    sa = ExtractedField(
        field_id="s1",
        field_type=src_type,
        label=src_label,
        value=src_value,
        normalized_label=src_label.lower(),
        normalized_value=src_value.lower(),
        bbox=_box(),
        confidence=1.0,
    )
    sb = ExtractedField(
        field_id="t1",
        field_type=tgt_type,
        label=tgt_label,
        value=tgt_value,
        normalized_label=tgt_label.lower(),
        normalized_value=tgt_value.lower(),
        bbox=_box(),
        confidence=1.0,
    )
    if match_type == MatchType.MISSING_IN_TARGET:
        return MatchResult(
            source_field=sa,
            target_field=None,
            match_type=match_type,
            confidence=confidence,
            reason=reason,
        )
    if match_type == MatchType.EXTRA_IN_TARGET:
        return MatchResult(
            source_field=None,
            target_field=sb,
            match_type=match_type,
            confidence=confidence,
            reason=reason,
        )
    return MatchResult(
        source_field=sa,
        target_field=sb,
        match_type=match_type,
        confidence=confidence,
        reason=reason,
    )


def test_missing_critical_is_high_severity() -> None:
    r = _pair(match_type=MatchType.MISSING_IN_TARGET, src_type="revision")
    cfg = DiscrepancyClassificationConfig()
    flags = discrepancy_flags_for_matches([r], cfg)
    assert len(flags) == 1
    assert flags[0].type == ReviewFlagType.MISSING_INFORMATION
    assert flags[0].severity == ReviewFlagSeverity.HIGH
    assert "[rule:missing_information]" in flags[0].reason
    assert "critical_field=True" in flags[0].reason


def test_missing_generic_medium_or_low() -> None:
    r = _pair(match_type=MatchType.MISSING_IN_TARGET, src_type="generic", src_label="")
    flags = discrepancy_flags_for_matches([r])
    assert flags[0].severity in (ReviewFlagSeverity.MEDIUM, ReviewFlagSeverity.LOW)


def test_extra_critical_medium_severity() -> None:
    r = _pair(match_type=MatchType.EXTRA_IN_TARGET, tgt_type="drawing_number")
    flags = discrepancy_flags_for_matches([r])
    assert flags[0].type == ReviewFlagType.EXTRA_INFORMATION
    assert flags[0].severity == ReviewFlagSeverity.MEDIUM


def test_changed_value_includes_value_similarity() -> None:
    r = _pair(match_type=MatchType.CHANGED_VALUE, src_value="alpha", tgt_value="omega")
    flags = discrepancy_flags_for_matches([r])
    types_ = [f.type for f in flags]
    assert ReviewFlagType.CHANGED_VALUE in types_
    ch = next(f for f in flags if f.type == ReviewFlagType.CHANGED_VALUE)
    assert "value_similarity=" in ch.reason
    assert "label_certainty=" in ch.reason


def test_uncertain_emits_ambiguous_match() -> None:
    r = _pair(match_type=MatchType.UNCERTAIN, confidence=0.35)
    flags = discrepancy_flags_for_matches([r])
    amb = [f for f in flags if f.type == ReviewFlagType.AMBIGUOUS_MATCH]
    assert len(amb) == 1
    assert amb[0].severity == ReviewFlagSeverity.HIGH
    err = [f for f in flags if f.type == ReviewFlagType.POSSIBLE_ERROR]
    assert len(err) == 1


def test_partial_match_emits_ambiguous() -> None:
    r = _pair(match_type=MatchType.PARTIAL_MATCH, confidence=0.85)
    flags = discrepancy_flags_for_matches([r])
    assert any(f.type == ReviewFlagType.AMBIGUOUS_MATCH for f in flags)
    assert not any(f.type == ReviewFlagType.POSSIBLE_ERROR for f in flags)


def test_exact_match_no_flags() -> None:
    r = _pair(
        match_type=MatchType.EXACT_MATCH,
        src_value="x",
        tgt_value="x",
        confidence=0.95,
    )
    flags = discrepancy_flags_for_matches([r])
    assert flags == []


def test_deterministic_order() -> None:
    r1 = _pair(match_type=MatchType.MISSING_IN_TARGET, src_type="generic")
    r1.source_field.field_id = "b"  # type: ignore[union-attr]
    r2 = _pair(match_type=MatchType.MISSING_IN_TARGET, src_type="generic")
    r2.source_field.field_id = "a"  # type: ignore[union-attr]
    f1 = discrepancy_flags_for_matches([r1, r2])
    f2 = discrepancy_flags_for_matches([r2, r1])
    assert [x.reason for x in f1] == [x.reason for x in f2]


def test_classifier_wrapper() -> None:
    dc = DiscrepancyClassifier()
    r = _pair(match_type=MatchType.EXTRA_IN_TARGET, tgt_type="generic")
    assert dc.classify([r])[0].type == ReviewFlagType.EXTRA_INFORMATION
