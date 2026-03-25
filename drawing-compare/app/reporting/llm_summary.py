"""Optional LLM narrative for a finalized :class:`~app.models.match.ComparisonReport`.

This module is **report-only**: it must not be imported from matching, classification,
or parsing. Input is strictly the structured ``ComparisonReport``; output is plain text.

Grounding checks reduce (but cannot eliminate) unsupported numeric claims and stray
match-type tokens. Callers should treat failures as soft-errors via
:attr:`LlmNarrativeSummaryOutcome.error` / grounding metadata.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from app.core.report_llm_summary_settings import ReportLlmSummaryConfig
from app.models.match import ComparisonReport, MatchType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You write short executive summaries of engineering drawing comparison reports. "
    "You will receive ONE JSON object: a ComparisonReport with summary counts and "
    "match rows. Rules:\n"
    "- Use only information that appears in that JSON. Do not invent fields, counts, "
    "drawing names, or match outcomes.\n"
    "- Every number you state must correspond to an integer that appears in the JSON "
    "text (not as part of a decimal like 0.88).\n"
    "- Do not introduce JSON keys or field names that are not present in the input.\n"
    "- 2–5 sentences, neutral tone, plain language.\n"
    "- Do not say you analyzed images or OCR; only summarize the structured report."
)


def comparison_report_canonical_json(report: ComparisonReport) -> str:
    """Stable JSON text for prompts and grounding (sorted keys, UTF-8 logical content)."""
    payload = report.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_non_decimal_integers(text: str) -> set[int]:
    """Integers not part of a ``<digits>.<digits>`` float token."""
    found: set[int] = set()
    for m in re.finditer(r"\d+", text):
        start, end = m.span()
        if end < len(text) and text[end] == "." and end + 1 < len(text) and text[end + 1].isdigit():
            continue
        if start > 0 and text[start - 1] == ".":
            continue
        found.add(int(m.group(), 10))
    return found


def _match_type_fragments_in_text(text: str) -> set[str]:
    present: set[str] = set()
    lowered = text.lower()
    for mt in MatchType:
        v = mt.value
        if v in text or v in lowered:
            present.add(v)
        # e.g. "exact match" vs exact_match
        spaced = v.replace("_", " ")
        if spaced.lower() in lowered:
            present.add(v)
    return present


def validate_summary_grounding(summary: str, canonical_json: str) -> list[str]:
    """Return human-readable violations; empty list means checks passed."""
    violations: list[str] = []
    summary_ints = extract_non_decimal_integers(summary)
    canonical_ints = extract_non_decimal_integers(canonical_json)
    for n in sorted(summary_ints):
        if n not in canonical_ints:
            violations.append(f"integer {n} appears in summary but not in report JSON integers")

    canon_types = _match_type_fragments_in_text(canonical_json)
    for token in sorted(_match_type_fragments_in_text(summary)):
        if token not in canon_types:
            violations.append(
                f"match outcome phrase tied to {token!r} appears in summary "
                "but that match type is not present in the report JSON",
            )
    return violations


def _chat_completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return f"{root}/chat/completions"


def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, object],
    timeout: float,
) -> dict[str, object]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "completions API returned non-object JSON"
        raise ValueError(msg)
    return parsed


def completion_usage_tokens(completion: dict[str, object]) -> dict[str, int] | None:
    """Parse OpenAI-style ``usage`` from a chat completions JSON object."""
    raw = completion.get("usage")
    if not isinstance(raw, dict):
        return None
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = raw.get(key)
        if v is None:
            continue
        try:
            out[key] = int(v)
        except (TypeError, ValueError):
            continue
    return out or None


def estimate_llm_cost_usd(
    usage: dict[str, int] | None,
    *,
    input_usd_per_million: float | None,
    output_usd_per_million: float | None,
) -> float | None:
    """Linear estimate from prompt + completion token counts; both prices must be set."""
    if usage is None or input_usd_per_million is None or output_usd_per_million is None:
        return None
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if pt is None or ct is None:
        return None
    raw = (pt / 1_000_000.0) * input_usd_per_million + (
        ct / 1_000_000.0
    ) * output_usd_per_million
    return round(raw, 6)


@dataclass(frozen=True)
class LlmNarrativeSummaryOutcome:
    """Result of a single summarization attempt (never used for matching logic)."""

    narrative: str | None
    error: str | None
    grounding_violations: tuple[str, ...]
    rejected_due_to_grounding: bool
    usage: dict[str, int] | None = None
    """Token counts from the completions API ``usage`` object, when present."""
    estimated_cost_usd: float | None = None
    """Rough USD cost when both per-million prices are configured on the summarizer."""


ChatCompletionPoster = Callable[
    [str, dict[str, str], dict[str, object], float],
    dict[str, object],
]


def _default_poster(
    url: str,
    headers: dict[str, str],
    body: dict[str, object],
    timeout: float,
) -> dict[str, object]:
    return _post_json(url, headers=headers, body=body, timeout=timeout)


def _parse_message_content(completion: dict[str, object]) -> str:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "completions response missing choices"
        raise ValueError(msg)
    first = choices[0]
    if not isinstance(first, dict):
        msg = "completions choice is not an object"
        raise ValueError(msg)
    msg_obj = first.get("message")
    if not isinstance(msg_obj, dict):
        msg = "completions choice missing message object"
        raise ValueError(msg)
    content = msg_obj.get("content")
    if not isinstance(content, str) or not content.strip():
        msg = "completions message content empty"
        raise ValueError(msg)
    return content.strip()


class ComparisonReportLlmSummarizer:
    """Optional summarizer; call only after the ComparisonReport is final."""

    def __init__(
        self,
        config: ReportLlmSummaryConfig,
        poster: ChatCompletionPoster | None = None,
    ) -> None:
        self._config = config
        self._poster = poster or _default_poster

    @property
    def model(self) -> str:
        """Chat completions model id from config (for usage logging)."""
        return self._config.model

    def summarize(self, report: ComparisonReport) -> LlmNarrativeSummaryOutcome:
        if not self._config.enabled:
            return LlmNarrativeSummaryOutcome(None, None, (), False)

        if not (self._config.api_key and self._config.api_key.strip()):
            return LlmNarrativeSummaryOutcome(
                None,
                "LLM summary enabled but report_llm_summary.api_key is not set",
                (),
                False,
            )

        canonical = comparison_report_canonical_json(report)
        if len(canonical) > self._config.max_report_json_chars:
            return LlmNarrativeSummaryOutcome(
                None,
                f"report JSON length {len(canonical)} exceeds max_report_json_chars "
                f"({self._config.max_report_json_chars})",
                (),
                False,
            )

        user_prompt = (
            "ComparisonReport JSON (only trustworthy source):\n\n"
            f"{canonical}\n\n"
            "Write the narrative summary per the system rules."
        )
        url = _chat_completions_url(self._config.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key.strip()}",
        }
        body: dict[str, object] = {
            "model": self._config.model,
            "max_completion_tokens": self._config.max_completion_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            completion = self._poster(url, headers, body, self._config.timeout_seconds)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:2000]
            logger.warning("LLM summary HTTP error %s: %s", e.code, err_body)
            return LlmNarrativeSummaryOutcome(
                None,
                f"LLM HTTP {e.code}: {err_body or e.reason}",
                (),
                False,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            logger.warning("LLM summary request failed: %s", e)
            return LlmNarrativeSummaryOutcome(None, f"LLM request failed: {e}", (), False)

        usage = completion_usage_tokens(completion)
        cost = estimate_llm_cost_usd(
            usage,
            input_usd_per_million=self._config.input_usd_per_million_tokens,
            output_usd_per_million=self._config.output_usd_per_million_tokens,
        )

        try:
            text = _parse_message_content(completion)
        except ValueError as e:
            return LlmNarrativeSummaryOutcome(
                None,
                str(e),
                (),
                False,
                usage=usage,
                estimated_cost_usd=cost,
            )

        violations = validate_summary_grounding(text, canonical)
        if violations:
            logger.info("LLM summary failed grounding: %s", violations)
            if self._config.reject_on_failed_grounding:
                return LlmNarrativeSummaryOutcome(
                    None,
                    "LLM summary rejected: failed grounding checks",
                    tuple(violations),
                    True,
                    usage=usage,
                    estimated_cost_usd=cost,
                )
            return LlmNarrativeSummaryOutcome(
                text,
                None,
                tuple(violations),
                False,
                usage=usage,
                estimated_cost_usd=cost,
            )

        return LlmNarrativeSummaryOutcome(
            text,
            None,
            (),
            False,
            usage=usage,
            estimated_cost_usd=cost,
        )


def summarizer_from_settings(
    config: ReportLlmSummaryConfig,
) -> ComparisonReportLlmSummarizer | None:
    """Factory: returns ``None`` when disabled so the builder skips LLM work entirely."""
    if not config.enabled:
        return None
    return ComparisonReportLlmSummarizer(config)
