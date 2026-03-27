"""Tests for :func:`app.ocr.normalizer.normalize_ocr_result`."""

from __future__ import annotations

from app.ocr.base import OCRResult
from app.ocr.normalizer import normalize_ocr_result


def test_normalize_renumbers_pages_and_text() -> None:
    raw = OCRResult(
        pages=[
            OCRResult.Page(
                page_number=3,
                width=100,
                height=200,
                lines=[
                    OCRResult.Line(
                        text="  hello   world  ",
                        bbox=(0, 0, 10, 10),
                        confidence=0.5,
                        tokens=[
                            OCRResult.Token(text="hello", bbox=(0, 0, 5, 10), confidence=0.5),
                            OCRResult.Token(text="world", bbox=(5, 0, 10, 10), confidence=0.5),
                        ],
                    )
                ],
            ),
            OCRResult.Page(
                page_number=1,
                width=50,
                height=50,
                lines=[],
            ),
        ]
    )
    out = normalize_ocr_result(raw)
    assert len(out.pages) == 2
    assert out.pages[0].page_number == 1
    assert out.pages[0].width == 50
    assert out.pages[1].page_number == 2
    assert out.pages[1].lines[0].text == "hello world"
    assert out.pages[1].lines[0].bbox == (0, 0, 10, 10)


def test_normalize_confidence_percent() -> None:
    # Raw API may emit percent scores; use model_construct to bypass OCRResult validation.
    raw = OCRResult.model_construct(
        pages=[
            OCRResult.Page.model_construct(
                page_number=1,
                width=10,
                height=10,
                lines=[
                    OCRResult.Line.model_construct(
                        text="x",
                        bbox=(0, 0, 2, 2),
                        confidence=50.0,
                        tokens=[
                            OCRResult.Token.model_construct(
                                text="x", bbox=(0, 0, 2, 2), confidence=80.0
                            )
                        ],
                    )
                ],
            )
        ]
    )
    out = normalize_ocr_result(raw)
    assert out.pages[0].lines[0].confidence == 0.8
    assert out.pages[0].lines[0].tokens[0].confidence == 0.8


def test_normalize_drops_empty_lines() -> None:
    raw = OCRResult(
        pages=[
            OCRResult.Page(
                page_number=1,
                width=10,
                height=10,
                lines=[
                    OCRResult.Line(
                        text="   ",
                        bbox=(0, 0, 1, 1),
                        confidence=0.0,
                        tokens=[],
                    ),
                    OCRResult.Line(
                        text="ok",
                        bbox=(0, 0, 2, 2),
                        confidence=1.0,
                        tokens=[OCRResult.Token(text="ok", bbox=(0, 0, 2, 2), confidence=1.0)],
                    ),
                ],
            )
        ]
    )
    out = normalize_ocr_result(raw)
    assert len(out.pages[0].lines) == 1
    assert out.pages[0].lines[0].text == "ok"
