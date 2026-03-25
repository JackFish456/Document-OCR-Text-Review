"""Settings model for optional ComparisonReport LLM narrative (no reporting imports)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReportLlmSummaryConfig(BaseModel):
    """LLM narrative settings (nested under application :class:`~app.core.config.Settings`)."""

    enabled: bool = Field(default=False, description="When true, builder may call the LLM.")
    api_key: str | None = Field(
        default=None,
        description=(
            "Bearer token for the chat Completions API (e.g. OpenAI). "
            "If unset, :class:`~app.core.config.Settings` may populate this from "
            "``DRAWING_COMPARE_OPENAI_API_KEY`` or ``OPENAI_API_KEY``."
        ),
        repr=False,
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API root; ``/chat/completions`` is appended.",
    )
    model: str = Field(default="gpt-4o-mini", description="Chat completions model id.")
    max_completion_tokens: int = Field(default=400, ge=32, le=8192)
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)
    max_report_json_chars: int = Field(
        default=100_000,
        ge=2_000,
        le=500_000,
        description="Refuse summarization when canonical JSON exceeds this size.",
    )
    reject_on_failed_grounding: bool = Field(
        default=True,
        description="If true, omit narrative when grounding checks fail.",
    )
