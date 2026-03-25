"""Run extractors over OCR pages / documents."""

from __future__ import annotations

from uuid import uuid4

from app.models.extraction import ExtractedField
from app.models.ocr import OCRDocument, OCRPage
from app.parsing.line_fields.base import LineFieldExtractor
from app.parsing.line_fields.context import LineExtractionContext
from app.parsing.line_fields.matches import ExtractionMatch
from app.parsing.line_fields.rules import default_line_extractors
from app.text_normalization import normalize_label, normalize_value


def extract_fields_from_line(
    ctx: LineExtractionContext,
    extractors: list[LineFieldExtractor],
    *,
    dedupe: bool = True,
) -> list[ExtractedField]:
    """Apply all extractors to a single line; optionally drop duplicate semantics."""
    matches: list[ExtractionMatch] = []
    for ex in extractors:
        matches.extend(ex.extract(ctx))

    lc = ctx.line_confidence
    if dedupe and matches:
        matches = _dedupe_matches(matches, lc)
    fields: list[ExtractedField] = []
    for m in matches:
        conf = m.combined_confidence(lc)
        fields.append(
            ExtractedField(
                field_id=f"{m.rule_id}-{uuid4().hex[:10]}",
                field_type=m.field_type,
                label=m.label,
                value=m.value,
                raw_text=m.raw_text,
                normalized_label=normalize_label(m.label).normalized,
                normalized_value=normalize_value(m.value).normalized,
                page_number=ctx.page_number,
                bbox=ctx.line.bbox,
                confidence=conf,
                extraction_rule=m.rule_id,
            )
        )
    return fields


def extract_fields_from_page(
    page: OCRPage,
    extractors: list[LineFieldExtractor] | None = None,
    *,
    dedupe: bool = True,
) -> list[ExtractedField]:
    ex = extractors if extractors is not None else default_line_extractors()
    out: list[ExtractedField] = []
    for line in page.lines:
        ctx = LineExtractionContext(page_number=page.page_number, line=line)
        out.extend(extract_fields_from_line(ctx, ex, dedupe=dedupe))
    return out


def extract_fields_from_document(
    doc: OCRDocument,
    extractors: list[LineFieldExtractor] | None = None,
    *,
    dedupe: bool = True,
) -> list[ExtractedField]:
    ex = extractors if extractors is not None else default_line_extractors()
    out: list[ExtractedField] = []
    for page in doc.pages:
        out.extend(extract_fields_from_page(page, ex, dedupe=dedupe))
    return out


def _dedupe_matches(
    matches: list[ExtractionMatch],
    line_confidence: float,
) -> list[ExtractionMatch]:
    """Keep highest-confidence match per normalized (label, value) pair."""
    scored: list[tuple[float, ExtractionMatch]] = []
    for m in matches:
        conf = m.combined_confidence(line_confidence)
        scored.append((conf, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[tuple[str, str, str]] = set()
    out: list[ExtractionMatch] = []
    for _c, m in scored:
        k = _match_key(m)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out


def _match_key(m: ExtractionMatch) -> tuple[str, str]:
    """Dedupe by normalized semantics so e.g. title_block vs generic_kv do not duplicate."""
    return (
        normalize_label(m.label).normalized,
        normalize_value(m.value).normalized,
    )
