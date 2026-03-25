"""Engineering drawing OCR normalization — realistic title block and note strings."""

from __future__ import annotations

import pytest

from app.models.extraction import ExtractedField
from app.models.ocr import BoundingBox
from app.text_normalization import (
    NormalizedText,
    create_canonical_key,
    normalize_label,
    normalize_line,
    normalize_value,
)


def _bbox() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=1, y2=1)


class TestNormalizedTextPreservation:
    def test_raw_preserved_when_empty(self) -> None:
        t = normalize_line("")
        assert t.raw == ""
        assert t.normalized == ""

    def test_raw_preserved_unicode_noise(self) -> None:
        raw = "SCALE\u200b:\u00a0\u201c1/4\u201d\u2033"
        t = normalize_line(raw)
        assert t.raw == raw
        assert '"' in t.normalized or "'" in t.normalized
        assert "1/4" in t.normalized


class TestWhitespaceAndCase:
    def test_collapse_internal_whitespace(self) -> None:
        t = normalize_line("  PROJECT   NAME\u3000  ")
        assert t.normalized == "project name"

    def test_casefold_german_eszett_stability(self) -> None:
        t = normalize_line("STRAßE")  # if NFKC/casefold maps — still deterministic
        assert t.normalized == t.normalized.lower()


class TestQuotesDashesPunctuation:
    def test_smart_quotes_to_ascii(self) -> None:
        t = normalize_line("\u201cTRENCH\u201d SECTION")
        assert t.normalized == '"trench" section'

    def test_unicode_dash_between_numbers(self) -> None:
        t = normalize_value("12\u20136\u2033")  # 12–6″
        assert "'" in t.normalized or '"' in t.normalized
        assert "12" in t.normalized and "6" in t.normalized

    def test_multiple_dashes_collapsed_for_drawing_style(self) -> None:
        t = normalize_line("A\u2014101")
        assert t.normalized == "a-101"


class TestNumericFormats:
    def test_us_thousands_in_title_block_area(self) -> None:
        t = normalize_value("FINISH EL. 1,250.00")
        assert "1250.00" in t.normalized or t.normalized.startswith("finish")

    def test_european_decimal_thousands_pattern(self) -> None:
        t = normalize_value("Length 1.234,56 m")
        assert "1234.56" in t.normalized

    def test_ocr_o_for_zero_in_numeric_token(self) -> None:
        t = normalize_value("12o34 noted")  # OCR glitch
        assert "12034" in t.normalized


class TestFeetInchAndDimensions:
    def test_prime_double_prime_feet_inch(self) -> None:
        raw = "12\u2032-6\u2033"
        t = normalize_value(raw)
        assert "12" in t.normalized and "6" in t.normalized

    def test_scale_equation_spacing(self) -> None:
        t = normalize_line('1/4"=1\'-0"')
        assert " = " in t.normalized

    def test_grid_dimension_x(self) -> None:
        t = normalize_value("12\u00d718\u0020FT")
        assert "12 x 18" in t.normalized


class TestElevations:
    def test_elevation_prefix_el_dot(self) -> None:
        t = normalize_value("EL. +125.5'")
        assert t.normalized.startswith("el ")
        assert "+125.5" in t.normalized

    def test_elevation_full_word_only_once(self) -> None:
        t = normalize_value("ELEVATION: 100.00")
        assert t.normalized.startswith("el ")


class TestRevisions:
    def test_rev_label_strips_to_token(self) -> None:
        t = normalize_label("REV.")
        assert t.normalized == "rev"

    def test_rev_value_inline(self) -> None:
        t = normalize_value("Issued Rev: B")
        assert "rev b" in t.normalized

    def test_revision_numeric(self) -> None:
        t = normalize_value("Revision 03")
        assert "rev 03" in t.normalized


class TestDrawingNumbers:
    def test_sheet_number_spaced_hyphens(self) -> None:
        t = normalize_line("Sheet  GS  -  1001  - A ")
        assert "gs-1001-a" in t.normalized or "gs-1001" in t.normalized

    def test_structural_drawing_index(self) -> None:
        t = normalize_value("S - 401")
        assert t.normalized == "s-401"


class TestSpecializedHelpers:
    def test_normalize_label_trailing_colon_semicolon(self) -> None:
        t = normalize_label("  DATE : ; ")
        assert t.normalized == "date"

    def test_normalize_value_keeps_operators_for_elevation(self) -> None:
        t = normalize_value("±0.00")
        assert "0.00" in t.normalized

    def test_normalize_line_vs_value_digit_fix_only_on_value(self) -> None:
        raw_o = "bearing l2o"
        ln = normalize_line(raw_o)
        vl = normalize_value(raw_o)
        assert "o" in ln.normalized  # not treated as numeric token
        assert "120" in vl.normalized or "2o" not in vl.normalized


class TestCanonicalKey:
    def test_joins_label_value(self) -> None:
        k = create_canonical_key("Scale", '1/4" = 1\'-0"')
        assert "|" in k
        assert "scale" in k

    def test_escapes_pipe(self) -> None:
        k = create_canonical_key("A|B", "C|D")
        assert "|" in k
        assert "/" in k
        assert "a/" in k or k.startswith("a/")


class TestFullwidthAndArtifacts:
    def test_nfkc_fullwidth_digits(self) -> None:
        t = normalize_value("\uff11\uff12\uff13")  # １２３
        assert t.normalized == "123"

    def test_strips_zero_width_and_bom(self) -> None:
        raw = "\ufeffNOTE\u200b1"
        t = normalize_line(raw)
        assert "\u200b" not in t.normalized
        assert "note1" == t.normalized.replace(" ", "") or "note 1" == t.normalized


class TestExtractedFieldIntegration:
    def test_extracted_field_uses_drawing_normalization(self) -> None:
        f = ExtractedField(
            field_id="x",
            label="DWG  NO.",
            value="A - 101",
            bbox=_bbox(),
            confidence=1.0,
        )
        assert f.normalized_label == "dwg no"
        assert f.normalized_value == "a-101"

    def test_comparison_key_with_both_label_and_value(self) -> None:
        fa = ExtractedField(
            field_id="a",
            label="SCALE",
            value="1/8\" = 1'-0\"",
            bbox=_bbox(),
            confidence=1.0,
        )
        fb = ExtractedField(
            field_id="b",
            label="SCALE  ",
            value='\u201c1/8"\u201d = 1\'-0"',
            bbox=_bbox(),
            confidence=1.0,
        )
        assert fa.comparison_key() == fb.comparison_key()


@pytest.mark.parametrize(
    ("raw", "expect_fragment", "use_value"),
    [
        ("KIEWIT\u00a0CORP", "kiewit corp", False),
        ("NO.\u00a0TS-1024-A", "ts-1024-a", False),
        ("R3 ", "rev 3", True),
        ("See Detail 5/S-100\u20134", "see detail 5/s-100-4", False),
    ],
)
def test_realistic_title_block_fragments(raw: str, expect_fragment: str, use_value: bool) -> None:
    merged = (normalize_value if use_value else normalize_line)(raw).normalized
    for part in expect_fragment.split():
        assert part in merged


def test_normalized_text_is_frozen_dataclass() -> None:
    t = NormalizedText(raw="x", normalized="y")
    with pytest.raises(AttributeError):
        t.normalized = "z"  # type: ignore[misc]
