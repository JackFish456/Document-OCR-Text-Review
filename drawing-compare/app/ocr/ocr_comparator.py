"""Compare two :class:`OCRResult` documents (e.g. local vs Google) without LLMs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from app.models.ocr import normalize_ocr_text
from app.ocr.base import OCRResult


class OCRComparisonMetrics(BaseModel):
    """Quantitative comparison of two OCR extractions (local vs Google)."""

    token_count_difference: int = Field(
        description="Local token count minus Google token count",
    )
    line_count_difference: int = Field(
        description="Local line count minus Google line count",
    )
    token_overlap_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Multiset Jaccard: intersection / union over normalized token strings",
    )
    average_confidence_difference: float = Field(
        description="Mean local token confidence minus mean Google token confidence",
    )
    unmatched_tokens_count: int = Field(
        ge=0,
        description="missing_tokens + extra_tokens (instance counts)",
    )
    missing_tokens: int = Field(
        ge=0,
        description="Token instances in local not matched by Google (multiset excess)",
    )
    extra_tokens: int = Field(
        ge=0,
        description="Token instances in Google not matched by local",
    )

    def summary(self) -> dict[str, Any]:
        """Compact dict for reporting (no LLM)."""
        return {
            "token_overlap": self.token_overlap_ratio,
            "missing_tokens": self.missing_tokens,
            "extra_tokens": self.extra_tokens,
            "confidence_delta": self.average_confidence_difference,
        }


def _flatten_tokens(result: OCRResult) -> list[tuple[str, float]]:
    """(normalized_text, confidence) for every token in document order."""
    out: list[tuple[str, float]] = []
    for page in result.pages:
        for line in page.lines:
            for tok in line.tokens:
                key = normalize_ocr_text(tok.text)
                if key:
                    out.append((key, float(tok.confidence)))
    return out


def _line_count(result: OCRResult) -> int:
    return sum(len(p.lines) for p in result.pages)


def compare_ocr(local_result: OCRResult, google_result: OCRResult) -> OCRComparisonMetrics:
    """Multiset token comparison: local is the reference, google is the hypothesis."""
    local_flat = _flatten_tokens(local_result)
    google_flat = _flatten_tokens(google_result)

    n_local = len(local_flat)
    n_google = len(google_flat)
    token_count_difference = n_local - n_google
    line_count_difference = _line_count(local_result) - _line_count(google_result)

    c_local = Counter(t[0] for t in local_flat)
    c_google = Counter(t[0] for t in google_flat)
    keys = set(c_local) | set(c_google)

    intersection = 0
    union = 0
    missing_tokens = 0
    extra_tokens = 0
    for k in keys:
        lc = c_local[k]
        gc = c_google[k]
        intersection += min(lc, gc)
        union += max(lc, gc)
        missing_tokens += max(0, lc - gc)
        extra_tokens += max(0, gc - lc)

    if union == 0:
        token_overlap_ratio = 1.0
    else:
        token_overlap_ratio = intersection / union

    conf_local = sum(t[1] for t in local_flat) / n_local if n_local else 0.0
    conf_google = sum(t[1] for t in google_flat) / n_google if n_google else 0.0
    average_confidence_difference = conf_local - conf_google

    unmatched_tokens_count = missing_tokens + extra_tokens

    return OCRComparisonMetrics(
        token_count_difference=token_count_difference,
        line_count_difference=line_count_difference,
        token_overlap_ratio=token_overlap_ratio,
        average_confidence_difference=average_confidence_difference,
        unmatched_tokens_count=unmatched_tokens_count,
        missing_tokens=missing_tokens,
        extra_tokens=extra_tokens,
    )
