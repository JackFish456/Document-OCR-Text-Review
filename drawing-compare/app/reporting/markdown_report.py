"""Markdown summaries for comparison results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.models.comparison import CompareResponse
from app.models.match import MatchType
from app.reporting.display import (
    field_caption,
    flag_type_plain,
    match_type_plain,
    severity_plain,
    summarize_pair,
)
from app.reporting.narrative import build_executive_summary

_EMBEDDED_REPORT_KEYS = frozenset({"comparison_json_report", "comparison_markdown_report"})


class AnnotatedFinding(Protocol):
    """Lightweight annotation protocol used by the reviewer-facing markdown table."""

    index: int
    match_type: MatchType
    source_text: str
    target_text: str
    confidence: float


def _esc_cell(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_esc_cell(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def _clean_extras(extras: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in extras.items() if k not in _EMBEDDED_REPORT_KEYS}


def build_comparison_markdown(
    response: CompareResponse,
    *,
    annotated_findings: Sequence[AnnotatedFinding] | None = None,
    annotated_pdf_href: str | None = None,
    annotated_visual_kind: str = "pdf",
) -> str:
    """Human-readable Markdown with an executive summary and sectioned detail."""
    report = response.report
    s = report.summary
    flags = response.review_flags
    clean_extras = _clean_extras(dict(response.extras))
    lines: list[str] = []

    lines.append("# Drawing comparison report")
    lines.append("")
    lines.append("## At a glance")
    lines.append("")
    lines.append(build_executive_summary(report, flags))
    lines.append("")
    if annotated_pdf_href:
        visual_label = "PDF" if annotated_visual_kind == "pdf" else "image"
        lines.append(
            f"**Open annotated {visual_label}:** [{annotated_pdf_href}]({annotated_pdf_href})"
        )
        lines.append("")

    lines.append("## Summary counts")
    lines.append("")
    lines.append(
        _md_table(
            ["Measure", "Count", "What it means"],
            [
                [
                    "Fields on source drawing",
                    str(s.total_source),
                    "How many fields we tried to match",
                ],
                ["Matched closely", str(s.matched), "Text lines that lined up well"],
                [
                    "Text changed",
                    str(s.changed),
                    "Same slot, different wording or values",
                ],
                [
                    "Missing on target",
                    str(s.missing),
                    "Present on source, not found on target",
                ],
                [
                    "Only on target",
                    str(s.extra_target),
                    "Found on target, not on source",
                ],
                [
                    "Unclear matches",
                    str(s.uncertain),
                    "System was unsure—needs a person",
                ],
            ],
        )
    )

    if annotated_findings is not None:
        lines.append("## Annotated findings")
        lines.append("")
        if not annotated_findings:
            lines.append("_No overlay-visible findings were generated._")
            lines.append("")
        else:
            rows = []
            for finding in annotated_findings:
                rows.append(
                    [
                        str(finding.index),
                        match_type_plain(finding.match_type),
                        str(finding.source_text or "—"),
                        str(finding.target_text or "—"),
                        f"{finding.confidence:.0%}",
                        (
                            f"[Open {annotated_visual_kind.upper()}]({annotated_pdf_href})"
                            if annotated_pdf_href
                            else "—"
                        ),
                    ]
                )
            lines.append(
                _md_table(
                    [
                        "Annotation #",
                        "Finding type",
                        "Source text",
                        "Target text",
                        "Confidence",
                        "Annotated visual",
                    ],
                    rows,
                )
            )

    def section_title(name: str) -> None:
        lines.append(f"## {name}")
        lines.append("")

    changed = [m for m in report.matches if m.match_type == MatchType.CHANGED_VALUE]
    section_title("Text that changed between drawings")
    if not changed:
        lines.append("_No changed values were flagged._")
        lines.append("")
    else:
        rows = []
        for m in changed:
            d = summarize_pair(m)
            rows.append(
                [
                    str(d.get("field_label") or "—"),
                    str(d.get("on_source_drawing") or "—"),
                    str(d.get("on_target_drawing") or "—"),
                    f"{float(d['match_confidence']):.0%}",
                ]
            )
        lines.append(
            _md_table(
                ["Field (if known)", "On source drawing", "On target drawing", "Confidence"],
                rows,
            )
        )

    section_title("Missing on the target drawing")
    if not report.missing:
        lines.append("_Nothing flagged as missing._")
        lines.append("")
    else:
        rows = []
        for m in report.missing:
            src = m.source_field
            assert src is not None
            rows.append([field_caption(src), src.field_type, f"{m.confidence:.0%}"])
        lines.append(_md_table(["What we saw on source", "Field type", "Confidence"], rows))

    section_title("Extra text on the target drawing")
    if not report.extra_target:
        lines.append("_Nothing flagged as extra._")
        lines.append("")
    else:
        rows = []
        for m in report.extra_target:
            tgt = m.target_field
            assert tgt is not None
            rows.append([field_caption(tgt), tgt.field_type, f"{m.confidence:.0%}"])
        lines.append(_md_table(["What we saw on target", "Field type", "Confidence"], rows))

    partial = [m for m in report.matches if m.match_type == MatchType.PARTIAL_MATCH]
    section_title("Partial matches (similar but not identical)")
    if not partial:
        lines.append("_No partial matches._")
        lines.append("")
    else:
        rows = []
        for m in partial:
            d = summarize_pair(m)
            rows.append(
                [
                    str(d.get("field_label") or "—"),
                    str(d.get("on_source_drawing") or "—"),
                    str(d.get("on_target_drawing") or "—"),
                    match_type_plain(m.match_type),
                    f"{m.confidence:.0%}",
                ]
            )
        lines.append(
            _md_table(
                ["Field (if known)", "Source", "Target", "Status", "Confidence"],
                rows,
            )
        )

    section_title("Unclear matches (please review)")
    if not report.uncertain:
        lines.append("_No uncertain matches._")
        lines.append("")
    else:
        rows = []
        for m in report.uncertain:
            d = summarize_pair(m)
            rows.append(
                [
                    str(d.get("on_source_drawing") or "—"),
                    str(d.get("on_target_drawing") or "—"),
                    f"{m.confidence:.0%}",
                ]
            )
        lines.append(_md_table(["Source text", "Target text", "Confidence"], rows))

    section_title("Review flags")
    if not flags:
        lines.append("_No review flags._")
        lines.append("")
    else:
        rows = []
        for f in flags:
            rows.append(
                [
                    flag_type_plain(f.type),
                    f.severity.value.title(),
                    severity_plain(f.severity),
                    (f.field_reference or "—"),
                    f.reason[:200] + ("…" if len(f.reason) > 200 else ""),
                ]
            )
        lines.append(
            _md_table(
                ["Topic", "Severity", "Guidance", "Related field", "Details"],
                rows,
            )
        )

    if clean_extras:
        lines.append("## Additional notes")
        lines.append("")
        lines.append("```")
        lines.append(str(clean_extras))
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"_Comparison ID: `{response.comparison_id}`_")
    lines.append("")

    return "\n".join(lines)
