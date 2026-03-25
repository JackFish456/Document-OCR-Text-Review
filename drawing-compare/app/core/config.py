"""Central configuration with environment overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.report_llm_summary_settings import ReportLlmSummaryConfig
from app.preprocessing.settings import PreprocessConfig


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"


class Settings(BaseSettings):
    """Application settings. Override via env: DRAWING_COMPARE_*."""

    model_config = SettingsConfigDict(
        env_prefix="DRAWING_COMPARE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    app_name: str = "drawing-compare"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Field(default_factory=_default_data_dir)

    # Matching / classification thresholds (0..1 unless noted)
    fuzzy_match_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    fuzzy_changed_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    uncertain_score_low: float = Field(default=0.55, ge=0.0, le=1.0)
    min_confidence_for_auto: float = Field(default=0.6, ge=0.0, le=1.0)
    spatial_max_distance_px: float = Field(default=80.0, ge=0.0)

    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)

    # OCR provider selection (see factory for aliases: paddle, windows, etc.)
    ocr_provider: str = "stub"

    # File-path OCR (:class:`app.ocr.document_provider.OCRProvider`)
    document_ocr_provider: str = "stub_document"
    tesseract_lang: str = "eng"
    tesseract_config: str = ""
    tesseract_psm: int | None = None
    paddle_lang: str = "en"
    paddle_use_angle_cls: bool = True
    paddle_use_gpu: bool = False

    # Evaluation
    golden_manifest_path: Path | None = None

    # When ``report_llm_summary.api_key`` is empty, used as the LLM Bearer token (OpenAI-style).
    # Env: ``DRAWING_COMPARE_OPENAI_API_KEY`` (this project) before ``OPENAI_API_KEY`` (common).
    openai_api_key: str | None = Field(default=None, repr=False)

    # Optional LLM narrative (final ComparisonReport only; never used for matching)
    report_llm_summary: ReportLlmSummaryConfig = Field(default_factory=ReportLlmSummaryConfig)

    @model_validator(mode="after")
    def _apply_llm_api_key_fallbacks(self) -> Self:
        """Copy project/OpenAI env API keys into ``report_llm_summary`` when nested key is blank."""
        r = self.report_llm_summary
        if (r.api_key or "").strip():
            return self
        for candidate in (
            (self.openai_api_key or "").strip(),
            (os.environ.get("OPENAI_API_KEY") or "").strip(),
        ):
            if candidate:
                self.report_llm_summary = r.model_copy(update={"api_key": candidate})
                break
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Clear settings cache (tests)."""
    get_settings.cache_clear()
