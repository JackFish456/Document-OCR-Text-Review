"""Structured reports for API and batch jobs."""

from app.reporting.builder import ComparisonReportBuilder
from app.reporting.display import (
    field_caption,
    flag_type_plain,
    match_type_plain,
    severity_plain,
)
from app.reporting.json_report import (
    build_comparison_json_document,
    comparison_report_json_str,
)
from app.reporting.markdown_report import build_comparison_markdown
from app.reporting.narrative import build_executive_summary

__all__ = [
    "ComparisonReportBuilder",
    "build_comparison_json_document",
    "build_comparison_markdown",
    "build_executive_summary",
    "comparison_report_json_str",
    "field_caption",
    "flag_type_plain",
    "match_type_plain",
    "severity_plain",
]
