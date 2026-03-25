"""Tests for optional ComparisonReport LLM narrative (grounding + builder wiring)."""

from __future__ import annotations

import json

from app.core.config import Settings
from app.models.extraction import ExtractedField
from app.models.match import ComparisonReport, MatchResult, MatchType
from app.models.ocr import BoundingBox
from app.reporting.builder import ComparisonReportBuilder
from app.reporting.llm_summary import (
    ComparisonReportLlmSummarizer,
    ReportLlmSummaryConfig,
    comparison_report_canonical_json,
    extract_non_decimal_integers,
    validate_summary_grounding,
)
from app.rules.review import ReviewRulesEngine


def _box() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=1, y2=1)


def _minimal_report() -> ComparisonReport:
    src = ExtractedField(
        field_id="s1",
        field_type="generic",
        label="A",
        value="1",
        bbox=_box(),
        confidence=1.0,
    )
    tgt = ExtractedField(
        field_id="t1",
        field_type="generic",
        label="A",
        value="2",
        bbox=_box(),
        confidence=0.88,
    )
    results = [
        MatchResult(
            source_field=src,
            target_field=tgt,
            match_type=MatchType.CHANGED_VALUE,
            confidence=0.88,
            reason="test",
        ),
    ]
    return ComparisonReport.from_flat_results(results, total_source=1)


def test_canonical_json_is_sorted_and_parseable() -> None:
    report = _minimal_report()
    s = comparison_report_canonical_json(report)
    data = json.loads(s)
    assert data["summary"]["total_source"] == 1
    assert list(data.keys()) == sorted(data.keys())


def test_extract_non_decimal_integers_skips_floats() -> None:
    assert extract_non_decimal_integers("confidence 0.88 and 3 items") == {3}
    assert extract_non_decimal_integers("page 2") == {2}


def test_grounding_rejects_unknown_integer() -> None:
    report = _minimal_report()
    canon = comparison_report_canonical_json(report)
    bad = "There were 999 fields changed."
    v = validate_summary_grounding(bad, canon)
    assert any("999" in x for x in v)


def test_grounding_allows_subset_counts() -> None:
    report = _minimal_report()
    canon = comparison_report_canonical_json(report)
    ok = "One field pair showed changed_value; total_source is 1."
    assert validate_summary_grounding(ok, canon) == []


def test_grounding_rejects_unknown_match_type_phrase() -> None:
    report = _minimal_report()
    canon = comparison_report_canonical_json(report)
    bad = "Several fields are missing_in_target."
    v = validate_summary_grounding(bad, canon)
    assert v


def test_summarizer_mock_poster_accepts_grounded_text() -> None:
    report = _minimal_report()
    canon = comparison_report_canonical_json(report)

    def poster(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        assert "/chat/completions" in url
        assert "Authorization" in headers
        _ = body, timeout
        narrative = (
            f"The report covers {report.summary.total_source} source field(s). "
            "One changed_value was recorded."
        )
        assert validate_summary_grounding(narrative, canon) == []
        return {
            "choices": [
                {"message": {"content": narrative}},
            ],
        }

    cfg = ReportLlmSummaryConfig(
        enabled=True,
        api_key="sk-test",
        base_url="https://example.invalid/v1",
    )
    summ = ComparisonReportLlmSummarizer(cfg, poster=poster)
    out = summ.summarize(report)
    assert out.narrative
    assert not out.error
    assert not out.grounding_violations


def test_summarizer_rejects_failed_grounding() -> None:
    report = _minimal_report()

    def poster(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        _ = url, headers, body, timeout
        return {
            "choices": [
                {"message": {"content": "Fully bogus: 999 uncertain matches."}},
            ],
        }

    cfg = ReportLlmSummaryConfig(enabled=True, api_key="x", reject_on_failed_grounding=True)
    summ = ComparisonReportLlmSummarizer(cfg, poster=poster)
    out = summ.summarize(report)
    assert out.narrative is None
    assert out.rejected_due_to_grounding
    assert out.grounding_violations


def test_builder_includes_llm_extra_when_configured() -> None:
    report = _minimal_report()
    canon = comparison_report_canonical_json(report)

    def poster(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        _ = url, headers, body, timeout
        narrative = (
            f"Summary: {report.summary.total_source} source field(s), "
            f"{report.summary.changed} changed_value."
        )
        assert validate_summary_grounding(narrative, canon) == []
        return {"choices": [{"message": {"content": narrative}}]}

    rules = ReviewRulesEngine(Settings())
    summ = ComparisonReportLlmSummarizer(
        ReportLlmSummaryConfig(enabled=True, api_key="k"),
        poster=poster,
    )
    builder = ComparisonReportBuilder(rules, llm_summarizer=summ)
    flat = report.all_results_flat()
    resp = builder.build(
        flat,
        total_source=report.summary.total_source,
        extras={"mode": "test"},
        include_text_reports=True,
    )
    assert resp.extras.get("comparison_llm_narrative_summary")
    assert "comparison_llm_summary_error" not in resp.extras


def test_builder_llm_runs_without_text_report_exports() -> None:
    report = _minimal_report()
    canon = comparison_report_canonical_json(report)

    def poster(
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        _ = url, headers, body, timeout
        narrative = (
            f"Summary: {report.summary.total_source} source field(s), "
            f"{report.summary.changed} changed_value."
        )
        assert validate_summary_grounding(narrative, canon) == []
        return {"choices": [{"message": {"content": narrative}}]}

    rules = ReviewRulesEngine(Settings())
    summ = ComparisonReportLlmSummarizer(
        ReportLlmSummaryConfig(enabled=True, api_key="k"),
        poster=poster,
    )
    builder = ComparisonReportBuilder(rules, llm_summarizer=summ)
    flat = report.all_results_flat()
    resp = builder.build(
        flat,
        total_source=report.summary.total_source,
        extras={"mode": "test"},
        include_text_reports=False,
    )
    assert resp.extras.get("comparison_llm_narrative_summary")
    assert "comparison_json_report" not in resp.extras
