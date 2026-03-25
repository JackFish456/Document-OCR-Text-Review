"""Derive review flags from comparison results and OCR quality."""

from __future__ import annotations

from app.classification.discrepancy_classifier import (
    DiscrepancyClassificationConfig,
    discrepancy_flags_for_matches,
)
from app.core.config import Settings
from app.models.match import MatchResult
from app.models.review_flag import ReviewFlag


class ReviewRulesEngine:
    """Delegates to :mod:`app.classification.discrepancy_classifier` (deterministic rules)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._discrepancy_config = DiscrepancyClassificationConfig.from_settings(settings)

    def flags_for_matches(self, results: list[MatchResult]) -> list[ReviewFlag]:
        return discrepancy_flags_for_matches(results, self._discrepancy_config)
