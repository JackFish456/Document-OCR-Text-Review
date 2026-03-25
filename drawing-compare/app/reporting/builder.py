"""Assemble comparison responses from classified results and flags."""

from __future__ import annotations

from typing import Any

from app.models.comparison import CompareResponse
from app.models.match import ComparisonReport, MatchResult
from app.reporting.json_report import comparison_report_json_str
from app.reporting.llm_summary import ComparisonReportLlmSummarizer
from app.reporting.markdown_report import build_comparison_markdown
from app.rules.review import ReviewRulesEngine


class ComparisonReportBuilder:
    """Single place to shape API payload and future export formats."""

    def __init__(
        self,
        rules: ReviewRulesEngine,
        *,
        llm_summarizer: ComparisonReportLlmSummarizer | None = None,
    ) -> None:
        self._rules = rules
        self._llm_summarizer = llm_summarizer

    def build(
        self,
        results: list[MatchResult],
        *,
        total_source: int,
        extras: dict[str, Any] | None = None,
        include_text_reports: bool = False,
    ) -> CompareResponse:
        report = ComparisonReport.from_flat_results(results, total_source=total_source)
        flags = self._rules.flags_for_matches(results)
        merged_extras: dict[str, Any] = dict(extras or {})
        response = CompareResponse(report=report, review_flags=flags, extras=merged_extras)
        if self._llm_summarizer is not None:
            llm_out = self._llm_summarizer.summarize(report)
            llm_extras = dict(response.extras)
            if llm_out.narrative:
                llm_extras["comparison_llm_narrative_summary"] = llm_out.narrative
            if llm_out.error:
                llm_extras["comparison_llm_summary_error"] = llm_out.error
            if llm_out.grounding_violations:
                llm_extras["comparison_llm_summary_grounding_violations"] = list(
                    llm_out.grounding_violations,
                )
            if llm_out.rejected_due_to_grounding:
                llm_extras["comparison_llm_summary_rejected_due_to_grounding"] = True
            if llm_out.usage is not None:
                usage_payload: dict[str, object] = {k: int(v) for k, v in llm_out.usage.items()}
                if llm_out.estimated_cost_usd is not None:
                    usage_payload["estimated_cost_usd"] = llm_out.estimated_cost_usd
                usage_payload["model"] = self._llm_summarizer.model
                llm_extras["comparison_llm_usage"] = usage_payload
            response = response.model_copy(update={"extras": llm_extras})
        if not include_text_reports:
            return response
        out_extras = {
            **response.extras,
            "comparison_json_report": comparison_report_json_str(response),
            "comparison_markdown_report": build_comparison_markdown(response),
        }
        return response.model_copy(update={"extras": out_extras})
