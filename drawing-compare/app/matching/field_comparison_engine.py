"""Field comparison: align source↔target fields and build a :class:`ComparisonReport`.

Scoring uses RapidFuzz on normalized label/value, plus token overlap and regex-style
value compatibility as fallbacks. Classification and confidence are derived from the
weighted composite and explicit threshold rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from rapidfuzz import fuzz

from app.matching.bipartite_assignment import optimal_assignment_max_weight
from app.models.extraction import ExtractedField
from app.models.match import ComparisonReport, MatchResult, MatchType
from app.vector_store.base import VectorCandidate

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_VALUE_PATTERN_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("ISO_DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("US_DATE", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("DIMENSION", re.compile(r"\b\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\b", re.IGNORECASE)),
    ("DECIMAL", re.compile(r"\b\d+\.\d+\b")),
    ("INTEGER", re.compile(r"\b\d+\b")),
)


@dataclass(frozen=True, slots=True)
class FieldComparisonConfig:
    """Tunable weights and thresholds for :class:`FieldComparisonEngine`."""

    # Weights for composite score (renormalized if label leg is omitted).
    weight_label: float = 0.45
    weight_value: float = 0.35
    weight_token_overlap: float = 0.12
    weight_regex_compat: float = 0.08
    # When vector similarity is applied, fuzzy weights are scaled by (1 - weight_vector).
    weight_vector: float = 0.25

    # Alignment: minimum composite to allow an edge in global one-to-one assignment.
    min_pair_composite: float = 0.38

    # When False, only source/target pairs on the same page_number are candidates.
    allow_cross_page_match: bool = False

    # Classification — label/value gates (0..1).
    exact_label_similarity: float = 0.92
    exact_value_similarity: float = 0.96
    strong_label_similarity: float = 0.82
    changed_value_max_value_similarity: float = 0.88
    partial_composite_min: float = 0.52
    uncertain_composite_max: float = 0.62

    @staticmethod
    def from_settings_like(
        *,
        fuzzy_match_threshold: float = 0.92,
        fuzzy_changed_threshold: float = 0.75,
        uncertain_score_low: float = 0.55,
        vector_weight: float = 0.25,
    ) -> FieldComparisonConfig:
        """Map legacy :class:`~app.core.config.Settings` fuzzy fields to engine thresholds."""
        return FieldComparisonConfig(
            exact_label_similarity=max(0.85, fuzzy_match_threshold),
            exact_value_similarity=max(0.9, fuzzy_match_threshold),
            strong_label_similarity=max(0.72, fuzzy_changed_threshold),
            changed_value_max_value_similarity=max(0.75, min(fuzzy_match_threshold, 0.9)),
            partial_composite_min=max(0.45, fuzzy_changed_threshold - 0.1),
            uncertain_composite_max=max(uncertain_score_low, 0.5),
            min_pair_composite=max(0.3, uncertain_score_low - 0.15),
            weight_vector=vector_weight,
        )


@dataclass(frozen=True, slots=True)
class PairScoreBreakdown:
    """Per-pair similarity signals in 0..1 (except reasons)."""

    label_similarity: float
    value_similarity: float
    token_overlap: float
    regex_compatibility: float
    vector_similarity: float | None
    composite_score: float
    reasons: tuple[str, ...]


def _normalized_label(f: ExtractedField) -> str:
    return (f.normalized_label or "").strip()


def _normalized_value(f: ExtractedField) -> str:
    return (f.normalized_value or "").strip()


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    r = max(
        float(fuzz.ratio(a, b)),
        float(fuzz.token_sort_ratio(a, b)),
    )
    return min(1.0, r / 100.0)


def _token_overlap_score(a: str, b: str) -> float:
    toks_a = set(_TOKEN_RE.findall(a.lower()))
    toks_b = set(_TOKEN_RE.findall(b.lower()))
    if not toks_a and not toks_b:
        return 1.0
    if not toks_a or not toks_b:
        return 0.0
    inter = toks_a & toks_b
    union = toks_a | toks_b
    jaccard = len(inter) / len(union)
    tsr = float(fuzz.token_set_ratio(a, b)) / 100.0
    return min(1.0, 0.5 * jaccard + 0.5 * tsr)


def _value_pattern_kind(s: str) -> str | None:
    for name, pat in _VALUE_PATTERN_RULES:
        if pat.search(s):
            return name
    return None


def regex_compatibility(value_a: str, value_b: str) -> float:
    """How compatible two value strings are under lightweight structural patterns."""
    ka = _value_pattern_kind(value_a)
    kb = _value_pattern_kind(value_b)
    if ka and kb:
        return 1.0 if ka == kb else 0.35
    if ka or kb:
        return 0.5
    return 0.75


def _effective_weights(
    *,
    label_a: str,
    label_b: str,
    cfg: FieldComparisonConfig,
) -> tuple[float, float, float, float]:
    w_l, w_v, w_t, w_r = (
        cfg.weight_label,
        cfg.weight_value,
        cfg.weight_token_overlap,
        cfg.weight_regex_compat,
    )
    if not label_a and not label_b:
        total = w_v + w_t + w_r
        return 0.0, w_v / total, w_t / total, w_r / total
    return w_l, w_v, w_t, w_r


def _composite_weights_with_vector(
    *,
    label_a: str,
    label_b: str,
    cfg: FieldComparisonConfig,
) -> tuple[float, float, float, float, float]:
    """Scale fuzzy weights by ``(1 - weight_vector)`` and reserve ``weight_vector`` for vector."""
    w_l, w_v, w_t, w_r = _effective_weights(label_a=label_a, label_b=label_b, cfg=cfg)
    w_vec = cfg.weight_vector
    if w_vec <= 0.0:
        return w_l, w_v, w_t, w_r, 0.0
    scale = 1.0 - w_vec
    return scale * w_l, scale * w_v, scale * w_t, scale * w_r, w_vec


def _target_matches_vector_candidate(candidate: VectorCandidate, target: ExtractedField) -> bool:
    pl = candidate.payload
    if pl.get("label") != target.label:
        return False
    if pl.get("value") != target.value:
        return False
    if pl.get("field_type") != target.field_type:
        return False
    page = pl.get("page")
    if page is not None and int(page) != int(target.page_number):
        return False
    return True


def _allowed_target_indices_for_source(
    src: ExtractedField,
    target_fields: list[ExtractedField],
) -> set[int] | None:
    """``None`` = no vector restriction; otherwise only these target indices are candidates."""
    if src.vector_candidates is None:
        return None
    allowed: set[int] = set()
    for j, tgt in enumerate(target_fields):
        for cand in src.vector_candidates:
            if _target_matches_vector_candidate(cand, tgt):
                allowed.add(j)
                break
    return allowed


def _vector_similarity_for_target(src: ExtractedField, tgt: ExtractedField) -> float | None:
    if src.vector_candidates is None:
        return None
    best: float | None = None
    for cand in src.vector_candidates:
        if _target_matches_vector_candidate(cand, tgt):
            s = float(cand.score)
            best = s if best is None else max(best, s)
    return best


def compute_pair_breakdown(
    source: ExtractedField,
    target: ExtractedField,
    cfg: FieldComparisonConfig,
    *,
    vector_similarity: float | None = None,
) -> PairScoreBreakdown:
    la, lb = _normalized_label(source), _normalized_label(target)
    va, vb = _normalized_value(source), _normalized_value(target)

    label_sim = _text_similarity(la, lb)
    value_sim = _text_similarity(va, vb)
    combined_for_tokens = f"{la} {va}".strip() or source.comparison_key()
    combined_for_tokens_t = f"{lb} {vb}".strip() or target.comparison_key()
    token_ov = _token_overlap_score(combined_for_tokens, combined_for_tokens_t)
    regex_c = regex_compatibility(va or source.value, vb or target.value)

    use_vector = vector_similarity is not None and cfg.weight_vector > 0.0
    vec_sim: float | None = float(vector_similarity) if use_vector else None

    if use_vector and vec_sim is not None:
        w_l, w_v, w_t, w_r, w_vec = _composite_weights_with_vector(
            label_a=la, label_b=lb, cfg=cfg
        )
        composite = (
            w_l * label_sim
            + w_v * value_sim
            + w_t * token_ov
            + w_r * regex_c
            + w_vec * vec_sim
        )
        reasons = (
            f"normalized_label_similarity={label_sim:.3f}",
            f"normalized_value_similarity={value_sim:.3f}",
            f"token_overlap={token_ov:.3f}",
            f"regex_compatibility={regex_c:.3f}",
            f"vector_similarity={vec_sim:.3f}",
            f"composite={composite:.3f} (weights label={w_l:.2f} value={w_v:.2f} "
            f"token={w_t:.2f} regex={w_r:.2f} vector={w_vec:.2f})",
        )
    else:
        w_l, w_v, w_t, w_r = _effective_weights(label_a=la, label_b=lb, cfg=cfg)
        composite = w_l * label_sim + w_v * value_sim + w_t * token_ov + w_r * regex_c
        reasons = (
            f"normalized_label_similarity={label_sim:.3f}",
            f"normalized_value_similarity={value_sim:.3f}",
            f"token_overlap={token_ov:.3f}",
            f"regex_compatibility={regex_c:.3f}",
            f"composite={composite:.3f} (weights label={w_l:.2f} value={w_v:.2f} "
            f"token={w_t:.2f} regex={w_r:.2f})",
        )

    return PairScoreBreakdown(
        label_similarity=label_sim,
        value_similarity=value_sim,
        token_overlap=token_ov,
        regex_compatibility=regex_c,
        vector_similarity=vec_sim,
        composite_score=composite,
        reasons=reasons,
    )


def _confidence_from(
    composite: float,
    match_type: MatchType,
    breakdown: PairScoreBreakdown,
) -> float:
    base = max(0.0, min(1.0, composite))
    if match_type == MatchType.EXACT_MATCH:
        return max(base, min(1.0, (breakdown.label_similarity + breakdown.value_similarity) / 2))
    if match_type == MatchType.CHANGED_VALUE:
        return max(0.0, min(1.0, 0.55 + 0.45 * breakdown.label_similarity))
    if match_type == MatchType.PARTIAL_MATCH:
        return max(0.0, min(1.0, 0.45 + 0.55 * base))
    if match_type == MatchType.UNCERTAIN:
        return max(0.0, min(1.0, 0.35 + 0.4 * base))
    return base


def _classify_paired(
    breakdown: PairScoreBreakdown,
    cfg: FieldComparisonConfig,
    *,
    source: ExtractedField,
    target: ExtractedField,
) -> MatchType:
    ls = breakdown.label_similarity
    vs = breakdown.value_similarity
    comp = breakdown.composite_score
    la = _normalized_label(source)
    lb = _normalized_label(target)
    labels_absent = not la and not lb

    if labels_absent:
        if vs >= cfg.exact_value_similarity:
            return MatchType.EXACT_MATCH
        if vs <= cfg.changed_value_max_value_similarity and comp >= cfg.partial_composite_min:
            return MatchType.CHANGED_VALUE
        if comp >= cfg.partial_composite_min:
            return MatchType.PARTIAL_MATCH
        if comp <= cfg.uncertain_composite_max:
            return MatchType.UNCERTAIN
        return MatchType.PARTIAL_MATCH

    if ls >= cfg.exact_label_similarity and vs >= cfg.exact_value_similarity:
        return MatchType.EXACT_MATCH
    if ls >= cfg.strong_label_similarity and vs <= cfg.changed_value_max_value_similarity:
        return MatchType.CHANGED_VALUE
    if ls >= cfg.strong_label_similarity and vs < cfg.exact_value_similarity:
        return MatchType.PARTIAL_MATCH
    if comp >= cfg.partial_composite_min:
        return MatchType.PARTIAL_MATCH
    if comp <= cfg.uncertain_composite_max:
        return MatchType.UNCERTAIN
    return MatchType.UNCERTAIN


def _decision_reasons(match_type: MatchType, breakdown: PairScoreBreakdown) -> tuple[str, ...]:
    head = (f"classification={match_type.value}",)
    return head + breakdown.reasons


class FieldComparisonEngine:
    """Aligns fields with composite scoring and produces a :class:`ComparisonReport`."""

    def __init__(self, config: FieldComparisonConfig | None = None) -> None:
        self._cfg = config or FieldComparisonConfig()

    @property
    def config(self) -> FieldComparisonConfig:
        return self._cfg

    def _candidate_weight_matrix(
        self,
        source_fields: list[ExtractedField],
        target_fields: list[ExtractedField],
    ) -> tuple[list[list[float | None]], list[list[PairScoreBreakdown | None]]]:
        """Pairwise composite scores; ``None`` when the edge is not a candidate."""
        n = len(source_fields)
        m = len(target_fields)
        weight: list[list[float | None]] = [[None] * m for _ in range(n)]
        breakdown: list[list[PairScoreBreakdown | None]] = [[None] * m for _ in range(n)]
        for i, src in enumerate(source_fields):
            allowed_j = _allowed_target_indices_for_source(src, target_fields)
            for j, tgt in enumerate(target_fields):
                if (
                    not self._cfg.allow_cross_page_match
                    and src.page_number != tgt.page_number
                ):
                    continue
                if allowed_j is not None and j not in allowed_j:
                    continue
                vec_sim: float | None = None
                if allowed_j is not None:
                    vec_sim = _vector_similarity_for_target(src, tgt)
                    if vec_sim is None:
                        continue
                bd = compute_pair_breakdown(
                    src,
                    tgt,
                    self._cfg,
                    vector_similarity=vec_sim,
                )
                if bd.composite_score < self._cfg.min_pair_composite:
                    continue
                weight[i][j] = bd.composite_score
                breakdown[i][j] = bd
        return weight, breakdown

    def align_and_score(
        self,
        source_fields: list[ExtractedField],
        target_fields: list[ExtractedField],
    ) -> list[tuple[ExtractedField, ExtractedField, PairScoreBreakdown]]:
        if not source_fields or not target_fields:
            return []
        weight, breakdown = self._candidate_weight_matrix(source_fields, target_fields)
        pairs_idx = optimal_assignment_max_weight(
            len(source_fields), len(target_fields), weight
        )
        out: list[tuple[ExtractedField, ExtractedField, PairScoreBreakdown]] = []
        for i, j in pairs_idx:
            bd = breakdown[i][j]
            if bd is None:
                continue
            out.append((source_fields[i], target_fields[j], bd))
        return out

    def build_match_results(
        self,
        source_fields: list[ExtractedField],
        target_fields: list[ExtractedField],
    ) -> list[MatchResult]:
        pairs = self.align_and_score(source_fields, target_fields)
        matched_src = {s.field_id for s, _, _ in pairs}
        matched_tgt = {t.field_id for _, t, _ in pairs}
        results: list[MatchResult] = []

        for src, tgt, bd in pairs:
            mtype = _classify_paired(bd, self._cfg, source=src, target=tgt)
            conf = _confidence_from(bd.composite_score, mtype, bd)
            reason = "; ".join(_decision_reasons(mtype, bd))
            results.append(
                MatchResult(
                    source_field=src,
                    target_field=tgt,
                    match_type=mtype,
                    confidence=conf,
                    reason=reason,
                )
            )

        for src in source_fields:
            if src.field_id not in matched_src:
                results.append(
                    MatchResult(
                        source_field=src,
                        target_field=None,
                        match_type=MatchType.MISSING_IN_TARGET,
                        confidence=1.0,
                        reason=(
                            "No target field met min_pair_composite="
                            f"{self._cfg.min_pair_composite:.3f} for this source row"
                        ),
                    )
                )

        for tgt in target_fields:
            if tgt.field_id not in matched_tgt:
                results.append(
                    MatchResult(
                        source_field=None,
                        target_field=tgt,
                        match_type=MatchType.EXTRA_IN_TARGET,
                        confidence=1.0,
                        reason=(
                            "No source field aligned to this target row under global one-to-one "
                            "assignment (same composite thresholds)"
                        ),
                    )
                )

        return results

    def build_report(
        self,
        source_fields: list[ExtractedField],
        target_fields: list[ExtractedField],
    ) -> ComparisonReport:
        flat = self.build_match_results(source_fields, target_fields)
        return ComparisonReport.from_flat_results(flat, total_source=len(source_fields))


def compare_field_lists(
    source_fields: list[ExtractedField],
    target_fields: list[ExtractedField],
    *,
    config: FieldComparisonConfig | None = None,
) -> ComparisonReport:
    """Convenience: run :class:`FieldComparisonEngine` and return the report."""
    return FieldComparisonEngine(config).build_report(source_fields, target_fields)
