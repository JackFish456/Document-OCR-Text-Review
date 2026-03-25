"""Settings: LLM API key resolution from the process environment."""

from __future__ import annotations

import pytest

from app.core.config import Settings, reset_settings_cache
from app.core.report_llm_summary_settings import ReportLlmSummaryConfig


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_llm_api_key_prefers_nested_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-should-not-win")
    monkeypatch.setenv("DRAWING_COMPARE_OPENAI_API_KEY", "sk-dc-should-not-win")
    s = Settings(
        report_llm_summary=ReportLlmSummaryConfig(enabled=True, api_key="sk-nested"),
    )
    assert s.report_llm_summary.api_key == "sk-nested"


def test_llm_api_key_falls_back_to_openai_api_key_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(
        openai_api_key="sk-from-drawing-compare-field",
        report_llm_summary=ReportLlmSummaryConfig(enabled=True, api_key=None),
    )
    assert s.report_llm_summary.api_key == "sk-from-drawing-compare-field"


def test_llm_api_key_falls_back_to_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRAWING_COMPARE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai-env")
    s = Settings(report_llm_summary=ReportLlmSummaryConfig(enabled=True, api_key=""))
    assert s.report_llm_summary.api_key == "sk-from-openai-env"


def test_llm_api_key_from_drawing_compare_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DRAWING_COMPARE_OPENAI_API_KEY", "sk-dc-only")
    s = Settings(report_llm_summary=ReportLlmSummaryConfig(enabled=True, api_key=None))
    assert s.report_llm_summary.api_key == "sk-dc-only"


def test_llm_api_key_project_env_wins_over_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRAWING_COMPARE_OPENAI_API_KEY", "sk-dc")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    s = Settings(report_llm_summary=ReportLlmSummaryConfig(enabled=True, api_key=None))
    assert s.report_llm_summary.api_key == "sk-dc"


def test_llm_api_key_nested_settings_env_wins_over_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DRAWING_COMPARE_REPORT_LLM_SUMMARY__API_KEY", "sk-nested-env")
    s = Settings()
    assert s.report_llm_summary.api_key == "sk-nested-env"
