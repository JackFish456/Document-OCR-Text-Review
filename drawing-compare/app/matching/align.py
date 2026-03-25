"""Greedy many-to-many field alignment using RapidFuzz token sort ratio."""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from app.models.extraction import ExtractedField


@dataclass(frozen=True, slots=True)
class FieldCandidatePair:
    field_id_a: str
    field_id_b: str
    score: float  # 0-100 RapidFuzz


@dataclass(frozen=True, slots=True)
class FieldAlignment:
    """Maps fields from drawing A to B with best-effort pairs."""

    pairs: list[FieldCandidatePair]
    unmatched_a: list[str]
    unmatched_b: list[str]


def align_fields(fields_a: list[ExtractedField], fields_b: list[ExtractedField]) -> FieldAlignment:
    """Pair fields by highest fuzzy score; each field used at most once."""
    used_b: set[str] = set()
    pairs: list[FieldCandidatePair] = []

    for fa in fields_a:
        best: tuple[str, float] | None = None
        for fb in fields_b:
            if fb.field_id in used_b:
                continue
            score = float(fuzz.token_sort_ratio(fa.comparison_key(), fb.comparison_key()))
            if best is None or score > best[1]:
                best = (fb.field_id, score)
        if best is not None:
            used_b.add(best[0])
            pairs.append(FieldCandidatePair(field_id_a=fa.field_id, field_id_b=best[0], score=best[1]))

    matched_a = {p.field_id_a for p in pairs}
    matched_b = {p.field_id_b for p in pairs}
    unmatched_a = [f.field_id for f in fields_a if f.field_id not in matched_a]
    unmatched_b = [f.field_id for f in fields_b if f.field_id not in matched_b]
    return FieldAlignment(pairs=pairs, unmatched_a=unmatched_a, unmatched_b=unmatched_b)
