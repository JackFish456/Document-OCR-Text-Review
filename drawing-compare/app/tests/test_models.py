"""Core model validation and serialization."""

import pytest

from app.models.extraction import ExtractedField
from app.models.golden import GoldenPair, GoldenPairExpected
from app.models.match import MatchResult, MatchType
from app.models.ocr import BoundingBox, OCRDocument, OCRLine, OCRPage, OCRToken
from app.models.serialization import model_from_dict, model_from_json, model_to_dict, model_to_json


def test_bounding_box_validation_rejects_inverted_box() -> None:
    with pytest.raises(ValueError):
        BoundingBox(x1=1, y1=1, x2=0, y2=2)


def test_ocr_document_round_trip_json() -> None:
    tok = OCRToken(
        text="A1",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        line_id="L1",
        block_id="B1",
    )
    line = OCRLine(id="L1", bbox=tok.bbox, tokens=[tok])
    page = OCRPage(page_number=1, width=100, height=100, lines=[line])
    doc = OCRDocument(document_id="d1", provider_name="test", pages=[page])
    raw = model_to_json(doc, indent=None)
    restored = model_from_json(OCRDocument, raw)
    assert restored.document_id == "d1"
    assert restored.pages[0].tokens[0].text == "A1"


def test_match_result_dict_round_trip() -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=1, y2=1)
    src = ExtractedField(field_id="s", value="x", bbox=bbox, confidence=1.0)
    tgt = ExtractedField(field_id="t", value="x", bbox=bbox, confidence=1.0)
    mr = MatchResult(
        source_field=src,
        target_field=tgt,
        match_type=MatchType.EXACT_MATCH,
        confidence=1.0,
    )
    d = model_to_dict(mr)
    d.pop("match_key", None)
    out = model_from_dict(MatchResult, d)
    assert out.match_type == MatchType.EXACT_MATCH


def test_golden_pair() -> None:
    p = GoldenPair(
        pair_id="p1",
        source_file="a.png",
        target_file="b.png",
        expected=GoldenPairExpected.MATCHED,
    )
    assert p.expected == GoldenPairExpected.MATCHED
