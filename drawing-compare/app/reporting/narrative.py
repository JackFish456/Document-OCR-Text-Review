"""Plain-language executive summaries for comparison reports."""

from __future__ import annotations

from app.models.match import ComparisonReport, MatchType
from app.models.review_flag import ReviewFlag, ReviewFlagSeverity


def build_executive_summary(report: ComparisonReport, review_flags: list[ReviewFlag]) -> str:
    """Short overview for non-technical readers (2–4 sentences)."""
    s = report.summary
    partial = [m for m in report.matches if m.match_type == MatchType.PARTIAL_MATCH]
    high = sum(1 for f in review_flags if f.severity == ReviewFlagSeverity.HIGH)
    medium = sum(1 for f in review_flags if f.severity == ReviewFlagSeverity.MEDIUM)

    parts: list[str] = []

    opener = (
        f"This comparison looked at {s.total_source} text field(s) on the source drawing "
        "and compared them to the target drawing."
    )
    parts.append(opener)

    if s.matched and not (s.changed or s.missing or s.extra_target or s.uncertain):
        parts.append(
            f"All counted fields aligned cleanly ({s.matched} matched with no differences flagged)."
        )
    else:
        detail_bits: list[str] = []
        if s.matched:
            detail_bits.append(f"{s.matched} field(s) matched closely")
        if s.changed:
            detail_bits.append(f"{s.changed} showed different text")
        if s.missing:
            detail_bits.append(f"{s.missing} were missing on the target")
        if s.extra_target:
            detail_bits.append(f"{s.extra_target} appeared only on the target")
        if s.uncertain:
            detail_bits.append(f"{s.uncertain} need a closer look because the match was unclear")
        if partial:
            detail_bits.append(f"{len(partial)} were only a partial text match")
        if detail_bits:
            parts.append("Results: " + "; ".join(detail_bits) + ".")

    if review_flags:
        sev = f"{high} high-priority" if high else ""
        med = f"{medium} medium-priority" if medium else ""
        join_sev = " and ".join(x for x in (sev, med) if x)
        tail = (
            f"The system raised {len(review_flags)} review item(s)"
            + (f" ({join_sev})" if join_sev else "")
            + "—see the review list for specifics."
        )
        parts.append(tail)
    else:
        parts.append("No automated review items were raised for this run.")

    return " ".join(parts)
