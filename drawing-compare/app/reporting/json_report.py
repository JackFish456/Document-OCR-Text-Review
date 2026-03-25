"""Structured JSON documents for comparison results (machine + reader-friendly layers)."""

from __future__ import annotations

import json
from typing import Any

from app.models.comparison import CompareResponse
from app.models.match import ComparisonSummary, MatchType
from app.models.review_flag import ReviewFlag
from app.reporting.display import field_caption, flag_type_plain, severity_plain, summarize_pair
from app.reporting.narrative import build_executive_summary

_EMBEDDED_REPORT_KEYS = frozenset({"comparison_json_report", "comparison_markdown_report"})


def _extras_without_embedded_reports(extras: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extras.items() if k not in _EMBEDDED_REPORT_KEYS}


def _summary_counts_block(report_summary: ComparisonSummary) -> dict[str, Any]:
    s = report_summary
    return {
        "fields_on_source_drawing": s.total_source,
        "matched_closely": s.matched,
        "text_changed_between_drawings": s.changed,
        "missing_on_target_drawing": s.missing,
        "only_on_target_drawing": s.extra_target,
        "unclear_matches": s.uncertain,
        "technical": {
            "total_source": s.total_source,
            "matched": s.matched,
            "changed": s.changed,
            "missing": s.missing,
            "extra_target": s.extra_target,
            "uncertain": s.uncertain,
        },
    }


def _changed_rows(response: CompareResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in response.report.matches:
        if m.match_type != MatchType.CHANGED_VALUE:
            continue
        row = summarize_pair(m)
        row["what_this_means"] = (
            "The same type of field was found on both drawings, but the text does not match."
        )
        out.append(row)
    return out


def _partial_rows(response: CompareResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in response.report.matches:
        if m.match_type != MatchType.PARTIAL_MATCH:
            continue
        row = summarize_pair(m)
        row["what_this_means"] = (
            "Text is similar but not identical—confirm whether this is acceptable."
        )
        out.append(row)
    return out


def _missing_rows(response: CompareResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in response.report.missing:
        src = m.source_field
        assert src is not None
        out.append(
            {
                "what_this_means": "This text appears on the source drawing but no matching field "
                "was found on the target.",
                "on_source_drawing": field_caption(src),
                "field_type": src.field_type,
                "field_id": src.field_id,
                "match_confidence": round(m.confidence, 4),
                "technical_notes": m.reason,
            }
        )
    return out


def _extra_rows(response: CompareResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in response.report.extra_target:
        tgt = m.target_field
        assert tgt is not None
        out.append(
            {
                "what_this_means": "This text appears on the target drawing but did not match "
                "anything on the source.",
                "on_target_drawing": field_caption(tgt),
                "field_type": tgt.field_type,
                "field_id": tgt.field_id,
                "match_confidence": round(m.confidence, 4),
                "technical_notes": m.reason,
            }
        )
    return out


def _uncertain_rows(response: CompareResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in response.report.uncertain:
        row = summarize_pair(m)
        row["what_this_means"] = (
            "The system could not confidently decide if these fields belong together—"
            "please review manually."
        )
        out.append(row)
    return out


def _flag_rows(flags: list[ReviewFlag]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in flags:
        out.append(
            {
                "category": flag_type_plain(f.type),
                "category_code": f.type.value,
                "severity": f.severity.value,
                "severity_explained": severity_plain(f.severity),
                "field_reference": f.field_reference,
                "details": f.reason,
                "confidence": round(f.confidence, 4),
            }
        )
    return out


def build_comparison_json_document(response: CompareResponse) -> dict[str, Any]:
    """Full structured report: narrative sections plus embedded canonical comparison payload."""
    exec_summary = build_executive_summary(response.report, response.review_flags)
    structured = response.model_dump(mode="json")
    if isinstance(structured.get("extras"), dict):
        structured["extras"] = _extras_without_embedded_reports(structured["extras"])
    clean_extras = _extras_without_embedded_reports(dict(response.extras))
    return {
        "schema_version": "1.0",
        "comparison_id": str(response.comparison_id),
        "executive_summary": exec_summary,
        "summary_counts": _summary_counts_block(response.report.summary),
        "changed_values": _changed_rows(response),
        "partial_matches": _partial_rows(response),
        "missing_fields": _missing_rows(response),
        "extra_fields": _extra_rows(response),
        "uncertain_matches": _uncertain_rows(response),
        "review_flags": _flag_rows(response.review_flags),
        "structured_comparison": structured,
        "extras": clean_extras,
    }


def comparison_report_json_str(response: CompareResponse, *, indent: int = 2) -> str:
    """Serialize :func:`build_comparison_json_document` to a UTF-8 JSON string."""
    doc = build_comparison_json_document(response)
    return json.dumps(doc, indent=indent, ensure_ascii=False)
