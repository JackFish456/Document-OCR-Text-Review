"""Assign golden-vs-predicted mismatches to OCR / extraction / matching / classification buckets."""

from __future__ import annotations

from app.evaluation.metrics import prediction_matches_golden, predicted_coarse_bucket
from app.models.diagnostics import PipelineDiagnostics
from app.models.error_analysis import ErrorCategory, FieldFailureRecord
from app.models.golden import GoldenExpectedLabel
from app.models.match import MatchResult, MatchType


def _preview(text: str | None, max_len: int = 80) -> str | None:
    if not text:
        return None
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _min_field_confidence(pred: MatchResult) -> float | None:
    if pred.source_field is not None and pred.target_field is not None:
        return min(pred.source_field.confidence, pred.target_field.confidence)
    if pred.source_field is not None:
        return pred.source_field.confidence
    if pred.target_field is not None:
        return pred.target_field.confidence
    return None


def categorize_field_failure(
    pair_id: str,
    field_id: str,
    gold: GoldenExpectedLabel,
    pred: MatchResult | None,
    diagnostics: PipelineDiagnostics,
    *,
    ocr_confidence_threshold: float = 0.55,
) -> FieldFailureRecord | None:
    """Return a record when ``gold`` disagrees with the engine for this ``field_id``."""
    if pred is not None and prediction_matches_golden(pred.match_type, gold):
        return None

    pred_bucket = predicted_coarse_bucket(pred.match_type if pred is not None else None)

    src_ids = {f.field_id for f in diagnostics.source_fields}
    tgt_ids = {f.field_id for f in diagnostics.target_fields}
    in_src = field_id in src_ids
    in_tgt = field_id in tgt_ids

    pred_type_str = pred.match_type.value if pred is not None else None
    mconf = _min_field_confidence(pred) if pred is not None else None
    src_prev = (
        _preview(pred.source_field.value or pred.source_field.raw_text) if pred and pred.source_field else None
    )
    tgt_prev = (
        _preview(pred.target_field.value or pred.target_field.raw_text) if pred and pred.target_field else None
    )
    reason = pred.reason if pred is not None else None
    match_conf = pred.confidence if pred is not None else None

    def record(
        category: ErrorCategory,
        pattern: str,
        rationale: str,
    ) -> FieldFailureRecord:
        return FieldFailureRecord(
            pair_id=pair_id,
            field_id=field_id,
            category=category,
            failure_pattern=pattern,
            gold_expected=gold,
            predicted_match_type=pred_type_str,
            predicted_bucket=pred_bucket,
            match_confidence=match_conf,
            engine_reason=_preview(reason, 200),
            source_value_preview=src_prev,
            target_value_preview=tgt_prev,
            min_field_confidence=mconf,
            in_source_extraction=in_src,
            in_target_extraction=in_tgt,
            rationale=rationale,
        )

    if pred is None:
        rationale = (
            "No MatchResult row for this field_id — it was not present in extracted fields "
            "under the same id, so parsing never produced an alignable row."
        )
        if gold == GoldenExpectedLabel.EXTRA:
            if not in_tgt:
                pattern = "gold_extra_field_not_in_target_extraction"
            elif in_tgt:
                pattern = "gold_extra_field_in_target_but_no_result_row"
                rationale = (
                    "Target field was extracted but no EXTRA_IN_TARGET row references this id "
                    "(check field_id consistency with the matcher output)."
                )
        elif not in_src and gold != GoldenExpectedLabel.EXTRA:
            pattern = "gold_field_not_in_source_extraction"
        elif in_src:
            pattern = "gold_field_extracted_but_no_match_row"
            rationale = (
                "Field exists on source extraction list but no MatchResult references this field_id."
            )
        else:
            pattern = "gold_field_absent_from_extraction"
        return record(ErrorCategory.EXTRACTION, pattern, rationale)

    if pred.match_type == MatchType.UNCERTAIN:
        return record(
            ErrorCategory.MATCHING,
            f"gold_{gold.value}_pred_uncertain",
            "Greedy alignment produced a pair but composite scores landed in the uncertain band.",
        )

    if gold == GoldenExpectedLabel.MISSING and pred.match_type in (
        MatchType.EXACT_MATCH,
        MatchType.PARTIAL_MATCH,
        MatchType.CHANGED_VALUE,
    ):
        return record(
            ErrorCategory.MATCHING,
            "gold_missing_pred_paired",
            "Gold says the value is absent on B, but the matcher linked this source row to a target.",
        )

    if gold == GoldenExpectedLabel.EXTRA and pred.match_type.requires_both_fields():
        return record(
            ErrorCategory.MATCHING,
            "gold_extra_pred_paired",
            "Gold says target-only extra, but this field was paired with a source row.",
        )

    if gold in (GoldenExpectedLabel.MATCHED, GoldenExpectedLabel.CHANGED) and pred.match_type == (
        MatchType.MISSING_IN_TARGET
    ):
        return record(
            ErrorCategory.MATCHING,
            f"gold_{gold.value}_pred_missing_in_target",
            "Source field exists but no target candidate met min_pair_composite — alignment failure.",
        )

    if (
        mconf is not None
        and mconf < ocr_confidence_threshold
        and pred.source_field is not None
        and pred.target_field is not None
    ):
        return record(
            ErrorCategory.OCR,
            f"gold_{gold.value}_pred_{pred.match_type.value}_low_field_confidence",
            f"At least one side of the pair has field confidence below {ocr_confidence_threshold:.2f} "
            "(noisy OCR / weak extraction signal).",
        )

    if gold == GoldenExpectedLabel.MATCHED and pred.match_type == MatchType.CHANGED_VALUE:
        return record(
            ErrorCategory.CLASSIFICATION,
            "gold_matched_pred_changed_value",
            "Matcher linked fields but classified values as changed while gold treats them as equivalent.",
        )

    if gold == GoldenExpectedLabel.CHANGED and pred.match_type in (
        MatchType.EXACT_MATCH,
        MatchType.PARTIAL_MATCH,
    ):
        return record(
            ErrorCategory.CLASSIFICATION,
            f"gold_changed_pred_{pred.match_type.value}",
            "Matcher treated values as equivalent; gold expects a semantic change.",
        )

    return record(
        ErrorCategory.CLASSIFICATION,
        f"gold_{gold.value}_pred_{pred.match_type.value}",
        "Outcome mismatch after alignment; thresholds or similarity features likely need tuning.",
    )
