"""Human-readable labels and field captions for reports."""

from __future__ import annotations

from app.models.extraction import ExtractedField
from app.models.match import MatchResult, MatchType
from app.models.review_flag import ReviewFlagSeverity, ReviewFlagType

MATCH_TYPE_TITLE: dict[MatchType, str] = {
    MatchType.EXACT_MATCH: "Exact match",
    MatchType.PARTIAL_MATCH: "Partial match",
    MatchType.CHANGED_VALUE: "Changed text",
    MatchType.MISSING_IN_TARGET: "Missing on target",
    MatchType.EXTRA_IN_TARGET: "Only on target",
    MatchType.UNCERTAIN: "Unclear match",
}

FLAG_TYPE_TITLE: dict[ReviewFlagType, str] = {
    ReviewFlagType.MISSING_INFORMATION: "Missing information",
    ReviewFlagType.CHANGED_VALUE: "Text changed",
    ReviewFlagType.POSSIBLE_ERROR: "Possible read or pairing error",
    ReviewFlagType.AMBIGUOUS_MATCH: "Unclear or partial match",
    ReviewFlagType.EXTRA_INFORMATION: "Extra information on target",
}

SEVERITY_GUIDANCE: dict[ReviewFlagSeverity, str] = {
    ReviewFlagSeverity.HIGH: "High — review soon; may affect correctness.",
    ReviewFlagSeverity.MEDIUM: "Medium — worth confirming when you have time.",
    ReviewFlagSeverity.LOW: "Low — informational; often safe to ignore.",
}


def field_caption(f: ExtractedField) -> str:
    """One-line description of extracted text for reports."""
    label = (f.label or "").strip()
    value = (f.value or "").strip()
    raw = (f.raw_text or "").strip()
    if label and value:
        return f"{label}: {value}"
    if value:
        return value
    if label:
        return label
    if raw:
        return raw
    return "(no text captured)"


def match_type_plain(match_type: MatchType) -> str:
    return MATCH_TYPE_TITLE.get(match_type, match_type.value.replace("_", " "))


def flag_type_plain(t: ReviewFlagType) -> str:
    return FLAG_TYPE_TITLE.get(t, t.value.replace("_", " "))


def severity_plain(sev: ReviewFlagSeverity) -> str:
    return SEVERITY_GUIDANCE.get(sev, sev.value)


def format_confidence(value: float | None) -> str:
    """Human-readable confidence percentage; em dash when unavailable."""
    if value is None:
        return "—"
    return f"{value:.0%}"


def format_ocr_confidence_pair(
    source: ExtractedField | None,
    target: ExtractedField | None,
) -> str:
    """Display source/target OCR confidence without conflating it with match confidence."""
    src = format_confidence(source.confidence) if source is not None else None
    tgt = format_confidence(target.confidence) if target is not None else None
    if src and tgt:
        return f"{src} / {tgt}"
    return src or tgt or "—"


def summarize_pair(m: MatchResult) -> dict[str, str | float | None]:
    """Stable key-value row for JSON tables."""
    src = m.source_field
    tgt = m.target_field
    label_bits: list[str] = []
    if src is not None:
        label_bits.append((src.normalized_label or src.label or "").strip())
    if tgt is not None:
        label_bits.append((tgt.normalized_label or tgt.label or "").strip())
    field_label = next((b for b in label_bits if b), None)
    return {
        "field_label": field_label,
        "on_source_drawing": field_caption(src) if src else None,
        "on_target_drawing": field_caption(tgt) if tgt else None,
        "match_confidence": round(m.confidence, 4),
        "ocr_confidence": format_ocr_confidence_pair(src, tgt),
        "technical_match_type": m.match_type.value,
        "technical_notes": m.reason,
    }
