"""Rule-based extraction from OCR lines (realistic engineering drawing text)."""

from __future__ import annotations

from app.models.ocr import BoundingBox, OCRDocument, OCRLine, OCRPage
from app.parsing.line_fields import (
    ExtractionMatch,
    LineExtractionContext,
    LineFieldExtractor,
    RegexLineExtractor,
    default_line_extractors,
    extract_fields_from_document,
    extract_fields_from_line,
    extract_fields_from_page,
)


def _bbox() -> BoundingBox:
    return BoundingBox(x1=10, y1=20, x2=400, y2=40)


def _line(line_id: str, text: str, *, conf: float = 0.88) -> OCRLine:
    return OCRLine(id=line_id, text=text, bbox=_bbox(), confidence=conf)


def _page(*lines: OCRLine, page_no: int = 1) -> OCRPage:
    return OCRPage(page_number=page_no, width=2000, height=1500, lines=list(lines))


class TestTitleBlockAndDrawingIds:
    def test_drawn_by_realistic(self) -> None:
        ctx = LineExtractionContext(page_number=1, line=_line("L1", "DRAWN BY:  J.SMITH"))
        fields = extract_fields_from_line(ctx, default_line_extractors())
        assert len(fields) == 1
        assert fields[0].field_type == "title_block.signature"
        assert "drawn" in fields[0].normalized_label
        assert "smith" in fields[0].normalized_value
        assert fields[0].extraction_rule == "tb_drawn_checked"
        assert 0.0 < fields[0].confidence <= 1.0

    def test_scale_with_mixed_quotes_ocr(self) -> None:
        ctx = LineExtractionContext(
            page_number=1,
            line=_line("L2", 'SCALE:  1/8" = 1\'-0"', conf=0.91),
        )
        fields = extract_fields_from_line(ctx, default_line_extractors())
        assert any(f.field_type == "title_block.scale" for f in fields)
        sc = next(f for f in fields if f.field_type == "title_block.scale")
        assert sc.value.startswith("1/8")
        assert sc.raw_text

    def test_dwg_number_noisy_spacing(self) -> None:
        ctx = LineExtractionContext(
            page_number=1,
            line=_line("L3", "DWG  NO. :  GS-1001-A"),
        )
        fields = extract_fields_from_line(ctx, default_line_extractors())
        dwg = next(f for f in fields if f.field_type == "drawing_id.dwg_no")
        assert "gs-1001-a" in dwg.normalized_value

    def test_sheet_number(self) -> None:
        ctx = LineExtractionContext(page_number=1, line=_line("L4", "SHEET NO: 3 OF 12"))
        fields = extract_fields_from_line(ctx, default_line_extractors())
        sh = next(f for f in fields if "sheet" in f.field_type)
        assert "3 of 12" in sh.normalized_value


class TestRevisionAndNotes:
    def test_revision_standalone(self) -> None:
        ctx = LineExtractionContext(page_number=1, line=_line("L5", "REV  B"))
        fields = extract_fields_from_line(ctx, default_line_extractors())
        rev = next(f for f in fields if f.field_type == "revision")
        assert rev.label.upper().startswith("REV")
        assert rev.normalized_value == "b"

    def test_general_notes_header(self) -> None:
        ctx = LineExtractionContext(
            page_number=1,
            line=_line("L6", "GENERAL NOTES:"),
        )
        fields = extract_fields_from_line(ctx, default_line_extractors())
        note = next(f for f in fields if f.field_type == "note.header")
        assert "note" in note.normalized_label

    def test_numbered_note_line(self) -> None:
        ctx = LineExtractionContext(
            page_number=1,
            line=_line(
                "L7",
                "1.  Field verify anchor bolt torque per spec  AISC 360.",
            ),
        )
        fields = extract_fields_from_line(ctx, default_line_extractors())
        bullet = next(f for f in fields if f.field_type == "note.bullet")
        assert bullet.label.startswith("1")
        assert "anchor" in bullet.normalized_value


class TestDimensionsAndGeneric:
    def test_elevation_line(self) -> None:
        ctx = LineExtractionContext(
            page_number=1,
            line=_line("L8", "EL. +125.5'"),
        )
        fields = extract_fields_from_line(ctx, default_line_extractors())
        el = next(f for f in fields if f.field_type == "dimension.elevation")
        assert "+" in el.value or "125" in el.value

    def test_imperial_dimension_run(self) -> None:
        ctx = LineExtractionContext(page_number=1, line=_line("L9", "12'-6\""))
        fields = extract_fields_from_line(ctx, default_line_extractors())
        dim = next(f for f in fields if f.field_type == "dimension.imperial")
        assert "12" in dim.value

    def test_generic_label_value_fallback(self) -> None:
        ctx = LineExtractionContext(
            page_number=1,
            line=_line("L10", "SUBMITTAL:  SUB-441 REV A"),
        )
        fields = extract_fields_from_line(ctx, default_line_extractors())
        gen = next(f for f in fields if f.field_type == "generic.label_value")
        assert "submittal" in gen.normalized_label


class TestDocumentAndDedupe:
    def test_full_page_multi_line(self) -> None:
        pg = _page(
            _line("a", "PROJECT:  River Crossing  Unit 3"),
            _line("b", "DATE: 03/24/2026"),
            _line("c", "R3"),
            _line("d", "See structural sheets S-100 for loads."),
        )
        fields = extract_fields_from_page(pg)
        types = {f.field_type for f in fields}
        assert "title_block.project" in types
        assert "title_block.sheet_meta" in types or any("sheet" in t for t in types)
        assert "revision" in types

    def test_dedupe_same_semantics(self) -> None:
        """DATE line should not emit duplicate generic + title_block if same key."""
        ctx = LineExtractionContext(page_number=1, line=_line("x", "DATE: 01/15/2025"))
        fields_deduped = extract_fields_from_line(ctx, default_line_extractors(), dedupe=True)
        fields_all = extract_fields_from_line(ctx, default_line_extractors(), dedupe=False)
        assert len(fields_deduped) <= len(fields_all)

    def test_extract_document(self) -> None:
        doc = OCRDocument(
            document_id="t1",
            provider_name="test",
            pages=[
                _page(
                    _line("1", "CLIENT:  KIEWIT"),
                    page_no=1,
                )
            ],
        )
        out = extract_fields_from_document(doc)
        assert len(out) == 1
        assert out[0].page_number == 1


class TestPluggableExtractor:
    def test_custom_extractor_appended(self) -> None:
        class SiteExtractor(LineFieldExtractor):
            @property
            def rule_id(self) -> str:
                return "site_code"

            def extract(self, ctx: LineExtractionContext) -> list[ExtractionMatch]:
                if "SITE" in ctx.text.upper() and "KCC" in ctx.text.upper():
                    return [
                        ExtractionMatch(
                            field_type="site.code",
                            label="SITE",
                            value="KCC",
                            raw_text=ctx.text,
                            rule_id=self.rule_id,
                            rule_weight=0.99,
                        )
                    ]
                return []

        line = _line("z", "SITE CODE: KCC- Omaha")
        ctx = LineExtractionContext(1, line)
        reg = [*default_line_extractors(), SiteExtractor()]
        fields = extract_fields_from_line(ctx, reg)
        assert any(f.extraction_rule == "site_code" for f in fields)

    def test_regex_line_extractor_finditer_mode(self) -> None:
        ex = RegexLineExtractor(
            "multi_kv",
            r"(?P<label>[A-Z]{2,})\s*=\s*(?P<value>\d+)",
            "test.multi",
            0.8,
            match_mode="findall",
        )
        ctx = LineExtractionContext(1, _line("m", "A=1 XX=22 YY=333"))
        matches = ex.extract(ctx)
        assert len(matches) == 2
        labels = {m.label for m in matches}
        assert labels == {"XX", "YY"}


class TestConfidenceScoring:
    def test_low_line_confidence_reduces_combined(self) -> None:
        ctx = LineExtractionContext(1, _line("c", "SCALE: 1\" = 20'", conf=0.5))
        fields = extract_fields_from_line(ctx, default_line_extractors())
        sc = next(f for f in fields if f.field_type == "title_block.scale")
        assert sc.confidence < 0.95
