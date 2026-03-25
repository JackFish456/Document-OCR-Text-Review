"""
Engineering drawing OCR normalization: whitespace, punctuation, dimensions,
elevations, revisions, and drawing-style identifiers.
"""

from __future__ import annotations

import re
import unicodedata

from app.text_normalization.types import NormalizedText

# Characters commonly stripped or treated as noise from OCR / PDF extraction.
_CONTROL_AND_ARTIFACT_CHARS = re.compile(
    r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u00ad"
    r"\u200b-\u200f\u2028\u2029\ufeff\ufff0-\uffff]"
)

# Unicode space / line sep replacements → regular space
_SPACE_LIKE = dict.fromkeys(
    map(
        ord,
        (
            "\u00a0",  # nbsp
            "\u1680",
            "\u2000",
            "\u2001",
            "\u2002",
            "\u2003",
            "\u2004",
            "\u2005",
            "\u2006",
            "\u2007",
            "\u2008",
            "\u2009",
            "\u200a",
            "\u202f",
            "\u205f",
            "\u3000",
        ),
    ),
    ord(" "),
)

_SQ = chr(39)
_DQ = chr(34)
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": _SQ,
        "\u2019": _SQ,
        "\u201a": _SQ,
        "\u201b": _SQ,
        "\u2032": _SQ,
        "\u02b9": _SQ,
        "\u0060": _SQ,
        "\u00b4": _SQ,
        "\u201c": _DQ,
        "\u201d": _DQ,
        "\u201e": _DQ,
        "\u201f": _DQ,
        "\u2033": _DQ,
        "\u00ab": _DQ,
        "\u00bb": _DQ,
        "\u275d": _DQ,
        "\u275e": _DQ,
    }
)

# Any unicode dash / minus → hyphen-minus for predictable downstream parsing
_DASH_RUN = re.compile(
    r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d\u02d7\u0320]+"
)

_MULTI_SPACE = re.compile(r"\s+")

# US-style thousands groups: 1,234 or 12,345,678 optional decimals
_THOUSANDS_US = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")

# European-style when comma is decimal separator (digit groups with dot then comma)
_EU_DECIMAL = re.compile(r"\b\d{1,3}(?:\.\d{3})*,\d+\b")

# Drawing / sheet numbering like "A - 101" or "GS - 1001"
_DRAWING_SEPARATED = re.compile(
    r"([A-Za-z]{1,10})\s*[-–—]\s*(\d{1,6}(?:\s*[-–—]\s*[A-Za-z0-9]{1,6})?)"
)

# Elevation lead-ins
_ELEV_PREFIX = re.compile(
    r"\b(?:el|elev|elevation)\b\.?\s*[:.]?\s*",
    re.IGNORECASE,
)

# Revision label / value patterns
_REV_LABEL = re.compile(r"^r\.?\s*ev(?:ision)?\.?\s*$", re.IGNORECASE)
_REV_VALUE_INLINE = re.compile(
    r"\b(?:revision|r\.?\s*ev(?:ision)?|rev)(?:\s*#)?\s*[:\-]?\s*([A-Za-z0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE,
)

# Scale bar style: 1/4" = 1'-0"
_SCALE_EQ = re.compile(r"\s*=\s*")

# Token that is almost purely numeric (allow OCR letter substitutions)
_NUMERICISH_TOKEN = re.compile(r"^[0-9OoIl|/.,+\-]+$")

_O_TO_0 = str.maketrans({"O": "0", "o": "0"})
_I_TO_1 = str.maketrans({"I": "1", "l": "1", "|": "1"})


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _strip_controls_and_artifacts(text: str) -> str:
    s = text.translate(_SPACE_LIKE)
    s = _CONTROL_AND_ARTIFACT_CHARS.sub("", s)
    return s


def _normalize_quotes(text: str) -> str:
    s = text.translate(_QUOTE_TRANSLATION)
    # OCR often emits doubled inch marks after translating curly quotes (e.g. "1/8"" scale)
    s = re.sub(r'(?<=[0-9/])"{2,}(?=[\s=$]|$)', _DQ, s)
    # Drop decorative opening quote before fractions (curly quotes around 1/8" scale)
    s = re.sub(r'(?<![0-9])"(\d/\d)', r"\1", s)
    return s


def _normalize_dashes(text: str) -> str:
    return _DASH_RUN.sub("-", text)


def _normalize_cross_for_dimensions(text: str) -> str:
    """Normalize multiplication `x` between numbers (plans, grids)."""
    return re.sub(
        r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)",
        r"\1 x \2",
        text,
        flags=re.IGNORECASE,
    )


def _normalize_feet_inch_compound(text: str) -> str:
    """
    Normalize compound imperial dimensions: optional feet portion + inches.
    Examples: 12'-6\", 12-6 (between numbers with dash) when followed by quote hint.
    """
    s = text
    # 12'6" or 12' 6" → 12' 6"
    s = re.sub(
        r"(\d)\s*'\s*(\d)",
        r"\1' \2",
        s,
    )
    # Inch mark immediately after digits: ensure space before '"' when foot also present in token
    s = re.sub(r"(\d)''", r'\1"', s)
    return s


def _normalize_scale_equation(text: str) -> str:
    """Normalize spaces around '=' in scale strings."""
    return _SCALE_EQ.sub(" = ", text)


def _normalize_us_thousands(text: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        return m.group(0).replace(",", "")

    return _THOUSANDS_US.sub(_repl, text)


def _normalize_european_decimal(text: str) -> str:
    """1.234,56 → 1234.56 (common on international sheet sets)."""

    def _repl(m: re.Match[str]) -> str:
        chunk = m.group(0)
        body, dec = chunk.rsplit(",", 1)
        return body.replace(".", "") + "." + dec

    return _EU_DECIMAL.sub(_repl, text)


def _normalize_drawing_number_spacing(text: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        return f"{left.lower()}-{right.replace(' ', '').replace('---', '-').replace('--', '-')}"

    return _DRAWING_SEPARATED.sub(_repl, text)


def _normalize_elevation_phrase(text: str) -> str:
    """EL. +125.5'-0\" style: normalize prefix to single ``el `` token."""

    def _repl(_m: re.Match[str]) -> str:
        return "el "

    return _ELEV_PREFIX.sub(_repl, text, count=1)


def _normalize_revision_phrases(text: str) -> str:
    """Rev 3 / REV A / Revision 02 → ``rev`` + value with stable spacing."""

    # Title-block shorthand "R3" / "r 12" (avoid matching inside longer tokens)
    s = re.sub(r"(?<![a-z0-9])r\s*(\d{1,3})\b", r"rev \1", text, flags=re.IGNORECASE)
    s = _REV_VALUE_INLINE.sub(r"rev \1", s)

    # Trailing lone REV / REVISION label (label normalize will trim)
    s = re.sub(
        r"\b(?:revision|rev)\s*#?\s*$",
        "rev",
        s,
        flags=re.IGNORECASE,
    )
    return s


def _scrub_duplicate_punctuation(text: str) -> str:
    """Collapse runs of periods or stray semicolons from OCR."""
    s = re.sub(r"\.{3,}", "…", text)  # ellipsis marker then normalize to three dots? keep simple
    s = s.replace("…", "...")
    s = re.sub(r";{2,}", ";", s)
    return s


def _fix_numericish_ocr_letters(text: str) -> str:
    """Within mostly-numeric tokens, map common OCR letter glitches to digits."""

    def _fix_token(tok: str) -> str:
        if not _NUMERICISH_TOKEN.match(tok):
            return tok
        # Don't touch pure fractions like 1/4
        if "/" in tok and re.match(r"^\d/\d$", tok.replace("O", "0")):
            return tok
        t = tok.translate(_O_TO_0)
        t = t.translate(_I_TO_1)
        return t

    return " ".join(_fix_token(p) for p in text.split())


def _base_normalize(text: str, *, apply_ocr_digit_fix: bool) -> str:
    """Shared pipeline through lowercase and whitespace."""
    s = _nfkc(text)
    s = _strip_controls_and_artifacts(s)
    s = _normalize_quotes(s)
    s = _normalize_dashes(s)
    s = _normalize_feet_inch_compound(s)
    s = _normalize_cross_for_dimensions(s)
    s = _normalize_scale_equation(s)
    s = _scrub_duplicate_punctuation(s)
    s = s.casefold()
    s = _MULTI_SPACE.sub(" ", s).strip()
    if apply_ocr_digit_fix:
        s = _fix_numericish_ocr_letters(s)
    return s


def _line_finishing(text: str) -> str:
    """Drawing-number spacing and light numeric normalization for any line."""
    s = _normalize_us_thousands(text)
    s = _normalize_european_decimal(s)
    s = _normalize_drawing_number_spacing(s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def normalize_line(text: str) -> NormalizedText:
    """
    Normalize a generic OCR text line (notes, general subtitles, multi-purpose strings).

    Applies unicode cleanup, quotes/dashes/feet-inch marks, lowercase, whitespace,
    US/European numeric forms, and drawing-number hyphen spacing.
    """
    raw = text
    if not raw:
        return NormalizedText(raw=raw, normalized="")
    s = _base_normalize(raw, apply_ocr_digit_fix=False)
    s = _line_finishing(s)
    return NormalizedText(raw=raw, normalized=s)


def normalize_label(text: str) -> NormalizedText:
    """
    Normalize title-block style labels (DATE, SHEET NO, REV, etc.).

    Strips trailing field punctuation and treats lone revision labels uniformly.
    """
    raw = text
    if not raw:
        return NormalizedText(raw=raw, normalized="")
    s = _base_normalize(raw, apply_ocr_digit_fix=False)
    s = _line_finishing(s)
    s = s.strip(" \t:;.—-")
    # "rev." / "revision" tokens collapsed
    if _REV_LABEL.match(s):
        s = "rev"
    s = _MULTI_SPACE.sub(" ", s).strip()
    return NormalizedText(raw=raw, normalized=s)


def normalize_value(text: str) -> NormalizedText:
    """
    Normalize field values: dimensions, elevations, revisions, drawing numbers, scales.

    Enables OCR digit/letter confusion fixes inside numeric-looking tokens.
    """
    raw = text
    if not raw:
        return NormalizedText(raw=raw, normalized="")
    s = _base_normalize(raw, apply_ocr_digit_fix=True)
    s = _normalize_elevation_phrase(s)
    s = _normalize_us_thousands(s)
    s = _normalize_european_decimal(s)
    s = _normalize_revision_phrases(s)
    s = _normalize_drawing_number_spacing(s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return NormalizedText(raw=raw, normalized=s)


def create_canonical_key(label: str, value: str) -> str:
    """
    Stable key for matching / deduplication from a label+value pair.

    Pipe characters in either side are escaped to avoid ambiguous joins.
    """
    ln = normalize_label(label).normalized.replace("|", "/")
    vn = normalize_value(value).normalized.replace("|", "/")
    if not ln and not vn:
        return ""
    if not ln:
        return vn
    if not vn:
        return ln
    return f"{ln}|{vn}"
