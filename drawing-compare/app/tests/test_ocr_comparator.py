"""Tests for :mod:`app.evaluation.ocr_comparator`."""

from __future__ import annotations

from app.ocr.ocr_comparator import compare_ocr
from app.ocr.base import OCRResult


def _page_with_tokens(*words: str) -> OCRResult.Page:
    toks = [
        OCRResult.Token(text=w, bbox=(0, 0, 10, 10), confidence=0.9) for w in words
    ]
    line = OCRResult.Line(
        text=" ".join(words),
        bbox=(0, 0, 10, 10),
        confidence=0.9,
        tokens=toks,
    )
    return OCRResult.Page(page_number=1, width=100, height=100, lines=[line])


def test_compare_ocr_identical() -> None:
    a = OCRResult(pages=[_page_with_tokens("hello", "world")])
    b = OCRResult(pages=[_page_with_tokens("hello", "world")])
    m = compare_ocr(a, b)
    assert m.token_overlap_ratio == 1.0
    assert m.missing_tokens == 0
    assert m.extra_tokens == 0
    assert m.unmatched_tokens_count == 0
    assert m.token_count_difference == 0
    assert m.line_count_difference == 0
    s = m.summary()
    assert s["token_overlap"] == 1.0
    assert s["missing_tokens"] == 0
    assert s["extra_tokens"] == 0


def test_compare_ocr_partial_overlap() -> None:
    local = OCRResult(pages=[_page_with_tokens("a", "b", "c")])
    google = OCRResult(pages=[_page_with_tokens("b", "c", "d")])
    m = compare_ocr(local, google)
    # intersection {b,c} = 2, union max counts: a:1,b:1,c:1,d:1 = 4
    assert m.missing_tokens == 1  # a
    assert m.extra_tokens == 1  # d
    assert m.unmatched_tokens_count == 2
    assert m.token_overlap_ratio == 2 / 4
