"""Multi-page drawing compare: page index alignment and page_number preservation."""

from __future__ import annotations

from pathlib import Path

import fitz

from app.core.config import Settings
from app.models.match import MatchType
from app.parsing.fields import RegionParser
from app.preprocessing.settings import PreprocessConfig
from app.reporting.builder import ComparisonReportBuilder
from app.rules.review import ReviewRulesEngine
from app.services.compare import DrawingCompareService


def _write_n_page_pdf(path: Path, *, n: int, width: float = 200.0, height: float = 100.0) -> None:
    doc = fitz.open()
    for i in range(n):
        page = doc.new_page(width=width, height=height)
        page.insert_text((24.0, 50.0), f"P{i + 1}", fontsize=12)
    doc.save(str(path))
    doc.close()


def _build_compare_service(settings: Settings) -> DrawingCompareService:
    rules = ReviewRulesEngine(settings)
    reporter = ComparisonReportBuilder(rules)
    return DrawingCompareService(settings=settings, parser=RegionParser(), reporter=reporter)


def test_echo_ocr_two_pages_preserves_distinct_page_numbers(tmp_path: Path) -> None:
    """Each preprocessed page gets stamped; fields do not all report page 1."""
    src = tmp_path / "src.pdf"
    tgt = tmp_path / "tgt.pdf"
    _write_n_page_pdf(src, n=2)
    _write_n_page_pdf(tgt, n=2)

    settings = Settings(
        ocr_provider="echo",
        preprocess=PreprocessConfig(enable_deskew=False),
    )
    svc = _build_compare_service(settings)
    _resp, artifacts = svc.compare_paths_with_artifacts(src, tgt, ocr_provider="echo")

    assert len(artifacts.source_pages) == 2
    assert len(artifacts.target_pages) == 2
    assert {p.page_number for p in artifacts.source_pages} == {1, 2}
    src_pages = {f.page_number for f in artifacts.source_fields}
    tgt_pages = {f.page_number for f in artifacts.target_fields}
    assert src_pages == {1, 2}
    assert tgt_pages == {1, 2}

    exact = [r for r in artifacts.results if r.match_type == MatchType.EXACT_MATCH]
    assert len(exact) == 2
    pairs: set[tuple[int, int]] = set()
    for r in exact:
        assert r.source_field is not None and r.target_field is not None
        pairs.add((r.source_field.page_number, r.target_field.page_number))
    assert pairs == {(1, 1), (2, 2)}


def test_source_extra_page_surfaces_missing_in_target(tmp_path: Path) -> None:
    """Source page 2 with no aligned target page yields a missing-on-target style result."""
    src = tmp_path / "src2.pdf"
    tgt = tmp_path / "tgt1.pdf"
    _write_n_page_pdf(src, n=2)
    _write_n_page_pdf(tgt, n=1)

    settings = Settings(
        ocr_provider="echo",
        preprocess=PreprocessConfig(enable_deskew=False),
    )
    svc = _build_compare_service(settings)
    _resp, artifacts = svc.compare_paths_with_artifacts(src, tgt, ocr_provider="echo")

    assert len(artifacts.source_pages) == 2
    assert len(artifacts.target_pages) == 1
    missing = [r for r in artifacts.results if r.match_type == MatchType.MISSING_IN_TARGET]
    assert len(missing) == 1
    assert missing[0].source_field is not None
    assert missing[0].source_field.page_number == 2


def test_target_extra_page_surfaces_extra_in_target(tmp_path: Path) -> None:
    """Target page 2 with no aligned source page yields extra-on-target."""
    src = tmp_path / "src1.pdf"
    tgt = tmp_path / "tgt2.pdf"
    _write_n_page_pdf(src, n=1)
    _write_n_page_pdf(tgt, n=2)

    settings = Settings(
        ocr_provider="echo",
        preprocess=PreprocessConfig(enable_deskew=False),
    )
    svc = _build_compare_service(settings)
    _resp, artifacts = svc.compare_paths_with_artifacts(src, tgt, ocr_provider="echo")

    assert len(artifacts.source_pages) == 1
    assert len(artifacts.target_pages) == 2
    extra = [r for r in artifacts.results if r.match_type == MatchType.EXTRA_IN_TARGET]
    assert len(extra) == 1
    assert extra[0].target_field is not None
    assert extra[0].target_field.page_number == 2
