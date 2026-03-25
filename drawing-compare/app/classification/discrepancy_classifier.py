"""Deterministic discrepancy → :class:`ReviewFlag` mapping from comparison results.

Each emitted flag includes a short ``[rule:…]`` prefix in ``reason`` for audit trails.
No probabilistic or LLM logic — only thresholds and string similarity (RapidFuzz).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.core.config import Settings
from app.models.extraction import ExtractedField
from app.models.match import MatchResult, MatchType
from app.models.review_flag import ReviewFlag, ReviewFlagSeverity, ReviewFlagType


@dataclass(frozen=True, slots=True)
class DiscrepancyClassificationConfig:
    """Tunable inputs for :func:`discrepancy_flags_for_matches`."""

    critical_field_types: frozenset[str] = frozenset(
        {
            "title_block",
            "revision",
            "sheet_number",
            "drawing_number",
            "scale",
        }
    )
    # Paired-match confidence bands (from comparison engine / classifier).
    confidence_high: float = 0.72
    confidence_medium: float = 0.55
    # 1 - normalized_value_similarity above these → stronger "change" signal.
    value_delta_high: float = 0.40
    value_delta_medium: float = 0.15
    # Label agreement below this → weaker semantic certainty (paired fields).
    label_certainty_medium_threshold: float = 0.88
    label_certainty_low_threshold: float = 0.72
    # Below this match confidence, emit a possible_error for aligned pairs.
    low_match_confidence_cutoff: float = 0.55
    # Below this, treat ambiguous_match as higher severity.
    ambiguous_confidence_high_severity: float = 0.42

    @staticmethod
    def from_settings(settings: Settings) -> DiscrepancyClassificationConfig:
        return DiscrepancyClassificationConfig(
            low_match_confidence_cutoff=settings.uncertain_score_low,
            ambiguous_confidence_high_severity=max(0.35, settings.uncertain_score_low - 0.12),
        )


def _fmt_rule(rule_id: str, parts: Iterable[str]) -> str:
    tail = "; ".join(p for p in parts if p)
    return f"[rule:{rule_id}] {tail}" if tail else f"[rule:{rule_id}]"


def _field_reference(r: MatchResult) -> str | None:
    if r.source_field is not None:
        return r.source_field.field_id
    if r.target_field is not None:
        return r.target_field.field_id
    return None


def _value_similarity(a: ExtractedField, b: ExtractedField) -> float:
    va = (a.normalized_value or "").strip()
    vb = (b.normalized_value or "").strip()
    if not va and not vb:
        return 1.0
    if not va or not vb:
        return 0.0
    return float(fuzz.ratio(va, vb)) / 100.0


def _label_certainty(a: ExtractedField, b: ExtractedField) -> float:
    la = (a.normalized_label or "").strip()
    lb = (b.normalized_label or "").strip()
    if la and lb:
        return float(fuzz.ratio(la, lb)) / 100.0
    if la or lb:
        return 0.38
    return 0.78


def _pick_severity_3(
    low: ReviewFlagSeverity,
    mid: ReviewFlagSeverity,
    high: ReviewFlagSeverity,
    *,
    score: float,
    t_mid: float,
    t_high: float,
) -> ReviewFlagSeverity:
    """``score`` above ``t_high`` → high; above ``t_mid`` → medium; else low."""
    if score >= t_high:
        return high
    if score >= t_mid:
        return mid
    return low


def _missing_information_flag(r: MatchResult, cfg: DiscrepancyClassificationConfig) -> ReviewFlag:
    src = r.source_field
    assert src is not None
    critical = src.field_type in cfg.critical_field_types
    has_label = bool((src.normalized_label or src.label or "").strip())
    sev = ReviewFlagSeverity.HIGH if critical else ReviewFlagSeverity.MEDIUM
    if not critical and not has_label:
        sev = ReviewFlagSeverity.LOW
    parts = [
        "Source field absent on target drawing",
        f"field_type={src.field_type}",
        f"critical_field={critical}",
        f"match_confidence={r.confidence:.3f}",
    ]
    return ReviewFlag(
        type=ReviewFlagType.MISSING_INFORMATION,
        severity=sev,
        field_reference=_field_reference(r),
        reason=_fmt_rule("missing_information", parts),
        confidence=r.confidence,
    )


def _extra_information_flag(r: MatchResult, cfg: DiscrepancyClassificationConfig) -> ReviewFlag:
    tgt = r.target_field
    assert tgt is not None
    critical = tgt.field_type in cfg.critical_field_types
    has_label = bool((tgt.normalized_label or tgt.label or "").strip())
    sev = ReviewFlagSeverity.MEDIUM if critical else ReviewFlagSeverity.LOW
    if not critical and not has_label:
        sev = ReviewFlagSeverity.LOW
    parts = [
        "Target field has no aligned source counterpart",
        f"field_type={tgt.field_type}",
        f"critical_field={critical}",
        f"match_confidence={r.confidence:.3f}",
    ]
    return ReviewFlag(
        type=ReviewFlagType.EXTRA_INFORMATION,
        severity=sev,
        field_reference=_field_reference(r),
        reason=_fmt_rule("extra_information", parts),
        confidence=r.confidence,
    )


def _changed_value_flag(r: MatchResult, cfg: DiscrepancyClassificationConfig) -> ReviewFlag:
    src, tgt = r.source_field, r.target_field
    assert src is not None and tgt is not None
    v_sim = _value_similarity(src, tgt)
    delta = 1.0 - v_sim
    l_cert = _label_certainty(src, tgt)
    # Strong label match + large value delta + decent match confidence → high severity.
    label_boost = 0.2 if l_cert >= cfg.label_certainty_medium_threshold else 0.0
    severity_score = delta * (0.45 + 0.35 * r.confidence) + label_boost
    sev = _pick_severity_3(
        ReviewFlagSeverity.LOW,
        ReviewFlagSeverity.MEDIUM,
        ReviewFlagSeverity.HIGH,
        score=severity_score,
        t_mid=cfg.value_delta_medium + 0.15,
        t_high=cfg.value_delta_high + 0.2,
    )
    if r.confidence < cfg.confidence_medium:
        if _severity_rank(sev) > _severity_rank(ReviewFlagSeverity.MEDIUM):
            sev = ReviewFlagSeverity.MEDIUM
    parts = [
        "Value differs between source and target under aligned labels",
        f"value_similarity={v_sim:.3f}",
        f"label_certainty={l_cert:.3f}",
        f"match_confidence={r.confidence:.3f}",
    ]
    if r.reason:
        parts.append(f"engine={r.reason!s}")
    return ReviewFlag(
        type=ReviewFlagType.CHANGED_VALUE,
        severity=sev,
        field_reference=_field_reference(r),
        reason=_fmt_rule("changed_value", parts),
        confidence=r.confidence,
    )


def _severity_rank(s: ReviewFlagSeverity) -> int:
    return {ReviewFlagSeverity.LOW: 0, ReviewFlagSeverity.MEDIUM: 1, ReviewFlagSeverity.HIGH: 2}[s]


def _ambiguous_match_flag(r: MatchResult, cfg: DiscrepancyClassificationConfig) -> ReviewFlag:
    src, tgt = r.source_field, r.target_field
    assert src is not None and tgt is not None
    v_sim = _value_similarity(src, tgt)
    l_cert = _label_certainty(src, tgt)
    risk = (1.0 - r.confidence) * 0.55 + (1.0 - min(v_sim, l_cert)) * 0.45
    sev = _pick_severity_3(
        ReviewFlagSeverity.LOW,
        ReviewFlagSeverity.MEDIUM,
        ReviewFlagSeverity.HIGH,
        score=risk,
        t_mid=0.38,
        t_high=0.58,
    )
    if (
        r.match_type == MatchType.UNCERTAIN
        and r.confidence <= cfg.ambiguous_confidence_high_severity
    ):
        sev = ReviewFlagSeverity.HIGH
    parts = [
        f"match_type={r.match_type.value}",
        f"match_confidence={r.confidence:.3f}",
        f"value_similarity={v_sim:.3f}",
        f"label_certainty={l_cert:.3f}",
    ]
    if r.reason:
        parts.append(f"engine={r.reason!s}")
    return ReviewFlag(
        type=ReviewFlagType.AMBIGUOUS_MATCH,
        severity=sev,
        field_reference=_field_reference(r),
        reason=_fmt_rule("ambiguous_match", parts),
        confidence=r.confidence,
    )


def _partial_match_flag(r: MatchResult, cfg: DiscrepancyClassificationConfig) -> ReviewFlag:
    """Partial alignment: review label/value agreement."""
    src, tgt = r.source_field, r.target_field
    assert src is not None and tgt is not None
    v_sim = _value_similarity(src, tgt)
    l_cert = _label_certainty(src, tgt)
    ambiguity = 1.0 - min(v_sim, l_cert)
    sev = _pick_severity_3(
        ReviewFlagSeverity.LOW,
        ReviewFlagSeverity.MEDIUM,
        ReviewFlagSeverity.HIGH,
        score=ambiguity + (1.0 - r.confidence) * 0.35,
        t_mid=0.42,
        t_high=0.62,
    )
    if l_cert < cfg.label_certainty_low_threshold:
        sev = max(sev, ReviewFlagSeverity.MEDIUM, key=_severity_rank)
    parts = [
        "Partial textual match; confirm semantic equivalence",
        f"value_similarity={v_sim:.3f}",
        f"label_certainty={l_cert:.3f}",
        f"match_confidence={r.confidence:.3f}",
    ]
    if r.reason:
        parts.append(f"engine={r.reason!s}")
    return ReviewFlag(
        type=ReviewFlagType.AMBIGUOUS_MATCH,
        severity=sev,
        field_reference=_field_reference(r),
        reason=_fmt_rule("partial_match_review", parts),
        confidence=r.confidence,
    )


def _possible_error_low_confidence(
    r: MatchResult,
    cfg: DiscrepancyClassificationConfig,
) -> ReviewFlag | None:
    if r.match_type in (MatchType.MISSING_IN_TARGET, MatchType.EXTRA_IN_TARGET):
        return None
    if r.confidence >= cfg.low_match_confidence_cutoff:
        return None
    src, tgt = r.source_field, r.target_field
    assert src is not None and tgt is not None
    v_sim = _value_similarity(src, tgt)
    l_cert = _label_certainty(src, tgt)
    gap = 1.0 - r.confidence
    sev = _pick_severity_3(
        ReviewFlagSeverity.LOW,
        ReviewFlagSeverity.MEDIUM,
        ReviewFlagSeverity.HIGH,
        score=gap,
        t_mid=0.22,
        t_high=0.38,
    )
    parts = [
        "Low match confidence; possible OCR noise or wrong alignment",
        f"match_type={r.match_type.value}",
        f"match_confidence={r.confidence:.3f}",
        f"value_similarity={v_sim:.3f}",
        f"label_certainty={l_cert:.3f}",
    ]
    return ReviewFlag(
        type=ReviewFlagType.POSSIBLE_ERROR,
        severity=sev,
        field_reference=_field_reference(r),
        reason=_fmt_rule("possible_error_low_confidence", parts),
        confidence=max(0.0, min(1.0, 1.0 - r.confidence)),
    )


def discrepancy_flags_for_matches(
    results: list[MatchResult],
    config: DiscrepancyClassificationConfig | None = None,
) -> list[ReviewFlag]:
    """Derive actionable review flags from flat :class:`MatchResult` rows."""
    cfg = config or DiscrepancyClassificationConfig()
    flags: list[ReviewFlag] = []
    for r in results:
        if r.match_type == MatchType.MISSING_IN_TARGET:
            flags.append(_missing_information_flag(r, cfg))
        elif r.match_type == MatchType.EXTRA_IN_TARGET:
            flags.append(_extra_information_flag(r, cfg))
        elif r.match_type == MatchType.CHANGED_VALUE:
            flags.append(_changed_value_flag(r, cfg))
        elif r.match_type == MatchType.UNCERTAIN:
            flags.append(_ambiguous_match_flag(r, cfg))
        elif r.match_type == MatchType.PARTIAL_MATCH:
            flags.append(_partial_match_flag(r, cfg))

        sup = _possible_error_low_confidence(r, cfg)
        if sup is not None:
            flags.append(sup)

    return _sort_flags_deterministically(flags)


def _sort_flags_deterministically(flags: list[ReviewFlag]) -> list[ReviewFlag]:
    return sorted(
        flags,
        key=lambda f: (
            f.field_reference or "",
            f.type.value,
            f.severity.value,
            f.reason,
        ),
    )


@dataclass
class DiscrepancyClassifier:
    """Thin stateful wrapper for tests and dependency injection."""

    config: DiscrepancyClassificationConfig = field(default_factory=DiscrepancyClassificationConfig)

    def classify(self, results: list[MatchResult]) -> list[ReviewFlag]:
        return discrepancy_flags_for_matches(results, self.config)
