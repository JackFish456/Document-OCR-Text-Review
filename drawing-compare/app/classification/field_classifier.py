"""Map fuzzy scores and presence to `MatchResult`."""

from __future__ import annotations

from app.core.config import Settings
from app.matching.align import FieldCandidatePair
from app.models.extraction import ExtractedField
from app.models.match import MatchResult, MatchType


def classify_pair(
    pair: FieldCandidatePair | None,
    field_a: ExtractedField | None,
    field_b: ExtractedField | None,
    settings: Settings,
) -> MatchResult:
    """Classify a single logical comparison (pair or singleton)."""
    if pair is None:
        if field_a is not None and field_b is None:
            return MatchResult(
                source_field=field_a,
                target_field=None,
                match_type=MatchType.MISSING_IN_TARGET,
                confidence=1.0,
                reason="Present on source drawing but no aligned field on target",
            )
        if field_b is not None and field_a is None:
            return MatchResult(
                source_field=None,
                target_field=field_b,
                match_type=MatchType.EXTRA_IN_TARGET,
                confidence=1.0,
                reason="Present on target drawing without source counterpart",
            )
        raise ValueError("classify_pair requires at least one field")

    if field_a is None or field_b is None:
        raise ValueError("Paired classification requires both fields")

    score_ratio = pair.score  # 0-100
    high = settings.fuzzy_match_threshold * 100.0
    mid = settings.fuzzy_changed_threshold * 100.0
    low = settings.uncertain_score_low * 100.0
    conf = max(0.0, min(1.0, score_ratio / 100.0))

    if field_a.comparison_key() == field_b.comparison_key():
        return MatchResult(
            source_field=field_a,
            target_field=field_b,
            match_type=MatchType.EXACT_MATCH,
            confidence=conf,
            reason="Identical normalized key text",
        )
    if score_ratio >= high:
        return MatchResult(
            source_field=field_a,
            target_field=field_b,
            match_type=MatchType.PARTIAL_MATCH,
            confidence=conf,
            reason="High fuzzy similarity but not identical",
        )
    if score_ratio >= mid:
        return MatchResult(
            source_field=field_a,
            target_field=field_b,
            match_type=MatchType.CHANGED_VALUE,
            confidence=conf,
            reason="Text differs beyond near-match threshold",
        )
    if score_ratio >= low:
        return MatchResult(
            source_field=field_a,
            target_field=field_b,
            match_type=MatchType.UNCERTAIN,
            confidence=conf,
            reason="Score in uncertain band; recommend manual review",
        )
    return MatchResult(
        source_field=field_a,
        target_field=field_b,
        match_type=MatchType.CHANGED_VALUE,
        confidence=conf,
        reason="Low similarity; treated as change or misalignment",
    )
