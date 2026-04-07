"""Visual diff manifest, renderer, and manual-run artifact tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import cv2
import fitz
import numpy as np

from app.models.comparison import CompareResponse
from app.models.extraction import ExtractedField
from app.models.match import ComparisonReport, MatchResult, MatchType
from app.models.ocr import BoundingBox
from app.preprocessing.types import PreprocessPageResult
from app.reporting.reviewer_bundle import build_reviewer_bundle
from app.reporting.visual_diff import (
    build_visual_diff_artifacts,
    build_visual_diff_manifest,
    plan_badge_placements,
    render_visual_diff_overlay,
    render_visual_diff_overlay_pdf,
)
from app.services.compare_artifacts import CompareArtifacts
from scripts.run_manual_compare import run_manual_compare


def _bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _field(
    field_id: str,
    text: str,
    bbox: BoundingBox,
    *,
    page_number: int = 1,
) -> ExtractedField:
    return ExtractedField(
        field_id=field_id,
        value=text,
        raw_text=text,
        bbox=bbox,
        confidence=0.95,
        page_number=page_number,
    )


def _page(
    image: np.ndarray,
    *,
    source_path: str = "memory.png",
    page_number: int = 1,
) -> PreprocessPageResult:
    h, w = image.shape[:2]
    return PreprocessPageResult(
        page_number=page_number,
        source_path=source_path,
        width=w,
        height=h,
        original_bgr=image.copy(),
        processed_bgr=image.copy(),
        deskew_angle_deg=0.0,
        tiles=[],
    )


def _response(
    results: list[MatchResult],
    *,
    source_path: str = "source.png",
    target_path: str = "target.png",
) -> CompareResponse:
    total_source = sum(1 for r in results if r.source_field)
    report = ComparisonReport.from_flat_results(results, total_source=total_source)
    return CompareResponse(
        report=report,
        extras={
            "source_path": source_path,
            "target_path": target_path,
            "ocr_provider": "stub",
        },
    )


def _compare_artifacts(
    results: list[MatchResult],
    *,
    source_image: np.ndarray,
    target_image: np.ndarray,
) -> CompareArtifacts:
    source_fields = [r.source_field for r in results if r.source_field is not None]
    target_fields = [r.target_field for r in results if r.target_field is not None]
    return CompareArtifacts(
        source_pages=[_page(source_image, source_path="source.png")],
        target_pages=[_page(target_image, source_path="target.png")],
        results=results,
        source_fields=source_fields,
        target_fields=target_fields,
    )


def _compare_artifacts_multipage(
    results: list[MatchResult],
    *,
    source_images: list[np.ndarray],
    target_images: list[np.ndarray],
) -> CompareArtifacts:
    source_fields = [r.source_field for r in results if r.source_field is not None]
    target_fields = [r.target_field for r in results if r.target_field is not None]
    return CompareArtifacts(
        source_pages=[
            _page(img, source_path="source.png", page_number=i + 1) for i, img in enumerate(source_images)
        ],
        target_pages=[
            _page(img, source_path="target.png", page_number=i + 1) for i, img in enumerate(target_images)
        ],
        results=results,
        source_fields=source_fields,
        target_fields=target_fields,
    )


def test_visual_manifest_filters_and_normalizes_non_exact_rows() -> None:
    src_exact = _field("s0", "same", _bbox(5, 5, 15, 15))
    tgt_exact = _field("t0", "same", _bbox(5, 5, 15, 15))
    src_changed = _field("s1", "old note", _bbox(10, 20, 30, 40))
    tgt_changed = _field("t1", "new note", _bbox(50, 10, 80, 20))
    src_missing = _field("s2", "gone", _bbox(20, 60, 40, 80))
    tgt_extra = _field("t3", "added", _bbox(110, 25, 150, 45))
    src_uncertain = _field("s4", "check", _bbox(60, 50, 90, 70))
    tgt_uncertain = _field("t4", "maybe", _bbox(140, 60, 180, 90))

    results = [
        MatchResult(
            source_field=src_exact,
            target_field=tgt_exact,
            match_type=MatchType.EXACT_MATCH,
            confidence=1.0,
        ),
        MatchResult(
            source_field=src_changed,
            target_field=tgt_changed,
            match_type=MatchType.CHANGED_VALUE,
            confidence=0.82,
        ),
        MatchResult(
            source_field=src_missing,
            target_field=None,
            match_type=MatchType.MISSING_IN_TARGET,
            confidence=1.0,
        ),
        MatchResult(
            source_field=None,
            target_field=tgt_extra,
            match_type=MatchType.EXTRA_IN_TARGET,
            confidence=1.0,
        ),
        MatchResult(
            source_field=src_uncertain,
            target_field=tgt_uncertain,
            match_type=MatchType.UNCERTAIN,
            confidence=0.48,
        ),
    ]

    artifacts = _compare_artifacts(
        results,
        source_image=np.zeros((100, 100, 3), dtype=np.uint8),
        target_image=np.zeros((100, 200, 3), dtype=np.uint8),
    )
    manifest = build_visual_diff_manifest(_response(results), artifacts)

    assert manifest.annotation_count == 4
    assert [ann.match_type for ann in manifest.annotations] == [
        MatchType.CHANGED_VALUE,
        MatchType.MISSING_IN_TARGET,
        MatchType.EXTRA_IN_TARGET,
        MatchType.UNCERTAIN,
    ]
    changed = manifest.annotations[0]
    assert changed.source_bbox_normalized is not None
    assert changed.target_bbox_normalized is not None
    assert changed.source_bbox_normalized.x1 == 0.1
    assert changed.overlay_bbox_normalized.x1 == 0.07
    missing = manifest.annotations[1]
    assert missing.overlay_bbox_normalized.x1 == 0.17
    extra = manifest.annotations[2]
    assert extra.overlay_bbox_normalized.x1 == 0.535


def test_render_visual_diff_anchors_changed_overlay_to_source_box() -> None:
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    target = np.zeros((50, 200, 3), dtype=np.uint8)
    target[10:20, 100:150] = (255, 255, 255)

    src = _field("s1", "old", _bbox(10, 10, 20, 20))
    tgt = _field("t1", "new", _bbox(100, 10, 150, 20))
    result = MatchResult(
        source_field=src,
        target_field=tgt,
        match_type=MatchType.CHANGED_VALUE,
        confidence=0.8,
    )
    artifacts = _compare_artifacts(results=[result], source_image=source, target_image=target)
    manifest = build_visual_diff_manifest(_response([result]), artifacts)

    rendered = render_visual_diff_overlay(manifest, source_image=source, target_image=target)

    assert rendered[14, 14].sum() > 0
    assert rendered[30, 60].sum() == 0
    assert rendered[5, 5].sum() == 0


def test_render_visual_diff_fills_missing_source_box() -> None:
    source = np.zeros((60, 60, 3), dtype=np.uint8)
    target = np.zeros((60, 60, 3), dtype=np.uint8)
    src = _field("s1", "missing", _bbox(10, 12, 24, 28))
    result = MatchResult(
        source_field=src,
        target_field=None,
        match_type=MatchType.MISSING_IN_TARGET,
        confidence=1.0,
    )
    artifacts = _compare_artifacts(results=[result], source_image=source, target_image=target)
    manifest = build_visual_diff_manifest(_response([result]), artifacts)

    rendered = render_visual_diff_overlay(manifest, source_image=source, target_image=target)

    assert rendered[20, 16].sum() > 0
    assert rendered[2, 2].sum() == 0


def test_build_visual_diff_artifacts_html_contains_expected_sections() -> None:
    source = np.zeros((40, 40, 3), dtype=np.uint8)
    target = np.zeros((40, 40, 3), dtype=np.uint8)
    target[5:15, 5:20] = (255, 255, 255)
    src = _field("s1", "old", _bbox(5, 5, 15, 15))
    tgt = _field("t1", "new", _bbox(5, 5, 20, 15))
    result = MatchResult(
        source_field=src,
        target_field=tgt,
        match_type=MatchType.CHANGED_VALUE,
        confidence=0.77,
    )
    artifacts = _compare_artifacts(results=[result], source_image=source, target_image=target)
    bundle = build_visual_diff_artifacts(_response([result]), artifacts)

    assert bundle.manifest.annotation_count == 1
    assert bundle.overlay_pdf.startswith(b"%PDF-")
    assert "Visual Overlay Diff Report" in bundle.html
    assert "data-toggle-type=\"changed_value\"" in bundle.html
    assert "Match confidence" in bundle.html
    assert "OCR confidence" in bundle.html


def test_multipage_visual_bundle_has_two_slices_and_multipage_html() -> None:
    """Multi-page manifests expose one slice per aligned pair; PDF/HTML follow."""
    # Stacked PNG overlay uses vstack; page rasters must share width for numpy concat.
    src1 = np.zeros((40, 40, 3), dtype=np.uint8)
    src2 = np.zeros((48, 40, 3), dtype=np.uint8)
    tgt1 = np.zeros((40, 40, 3), dtype=np.uint8)
    tgt2 = np.zeros((48, 40, 3), dtype=np.uint8)
    r1 = MatchResult(
        source_field=_field("s1", "a", _bbox(5, 5, 15, 15), page_number=1),
        target_field=_field("t1", "b", _bbox(5, 5, 15, 15), page_number=1),
        match_type=MatchType.CHANGED_VALUE,
        confidence=0.8,
    )
    r2 = MatchResult(
        source_field=_field("s2", "c", _bbox(2, 2, 10, 10), page_number=2),
        target_field=_field("t2", "d", _bbox(2, 2, 10, 10), page_number=2),
        match_type=MatchType.CHANGED_VALUE,
        confidence=0.75,
    )
    artifacts = _compare_artifacts_multipage(
        [r1, r2],
        source_images=[src1, src2],
        target_images=[tgt1, tgt2],
    )
    manifest = build_visual_diff_manifest(_response([r1, r2]), artifacts)
    assert len(manifest.pages) == 2
    assert manifest.pages[0].pair_index == 0
    assert manifest.pages[1].pair_index == 1
    assert manifest.pages[0].source_page_number == 1
    assert manifest.pages[1].source_page_number == 2

    bundle = build_visual_diff_artifacts(_response([r1, r2]), artifacts)
    assert "Visual Overlay Diff Report (multi-page)" in bundle.html
    assert "diffCanvas-0" in bundle.html and "diffCanvas-1" in bundle.html

    doc = fitz.open(stream=bundle.overlay_pdf, filetype="pdf")
    try:
        assert len(doc) == 2
    finally:
        doc.close()


def test_render_visual_diff_overlay_pdf_is_valid_pdf() -> None:
    source = np.zeros((80, 120, 3), dtype=np.uint8)
    src = _field("s1", "missing", _bbox(20, 25, 45, 50))
    result = MatchResult(
        source_field=src,
        target_field=None,
        match_type=MatchType.MISSING_IN_TARGET,
        confidence=1.0,
    )
    artifacts = _compare_artifacts(results=[result], source_image=source, target_image=source)
    manifest = build_visual_diff_manifest(_response([result]), artifacts)

    pdf_bytes = render_visual_diff_overlay_pdf(manifest, source_image=source)

    assert pdf_bytes.startswith(b"%PDF-")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        assert len(doc) == 1
        page = doc[0]
        assert int(page.rect.width) == 120
        assert int(page.rect.height) == 80
    finally:
        doc.close()


def test_build_reviewer_bundle_falls_back_to_png_when_pdf_generation_fails() -> None:
    source = np.zeros((40, 40, 3), dtype=np.uint8)
    target = np.zeros((40, 40, 3), dtype=np.uint8)
    result = MatchResult(
        source_field=_field("s1", "old", _bbox(5, 5, 15, 15)),
        target_field=_field("t1", "new", _bbox(5, 5, 20, 15)),
        match_type=MatchType.CHANGED_VALUE,
        confidence=0.77,
    )
    artifacts = _compare_artifacts(results=[result], source_image=source, target_image=target)
    with patch(
        "app.reporting.reviewer_bundle.render_visual_diff_overlay_pdf_pages",
        side_effect=RuntimeError("pdf broke"),
    ):
        bundle = build_reviewer_bundle(_response([result]), artifacts)

    assert bundle.visual_kind == "png"
    assert bundle.visual_filename == "visual_diff_overlay.png"
    assert bundle.visual_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert bundle.comparison_docx[:2] == b"PK"


def test_plan_badge_placements_prefers_outside_when_space_exists() -> None:
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    src = _field("s1", "missing", _bbox(20, 40, 40, 60))
    result = MatchResult(
        source_field=src,
        target_field=None,
        match_type=MatchType.MISSING_IN_TARGET,
        confidence=1.0,
    )
    artifacts = _compare_artifacts(results=[result], source_image=source, target_image=source)
    manifest = build_visual_diff_manifest(_response([result]), artifacts)

    placement = plan_badge_placements(manifest)[0]

    assert placement.placement == "outside_top"
    assert placement.y2 <= 40


def test_plan_badge_placements_avoids_overlap_for_nearby_boxes() -> None:
    source = np.zeros((120, 120, 3), dtype=np.uint8)
    results = [
        MatchResult(
            source_field=_field("s1", "one", _bbox(18, 44, 40, 66)),
            target_field=None,
            match_type=MatchType.MISSING_IN_TARGET,
            confidence=1.0,
        ),
        MatchResult(
            source_field=_field("s2", "two", _bbox(30, 44, 52, 66)),
            target_field=None,
            match_type=MatchType.MISSING_IN_TARGET,
            confidence=1.0,
        ),
    ]
    artifacts = _compare_artifacts(results=results, source_image=source, target_image=source)
    manifest = build_visual_diff_manifest(_response(results), artifacts)

    first, second = plan_badge_placements(manifest)

    assert first.placement.startswith("outside")
    assert second.placement.startswith("outside")
    assert (
        first.x2 <= second.x1
        or second.x2 <= first.x1
        or first.y2 <= second.y1
        or second.y2 <= first.y1
    )


def test_plan_badges_prefer_not_covering_highlight_boxes() -> None:
    source = np.zeros((120, 120, 3), dtype=np.uint8)
    results = [
        MatchResult(
            source_field=_field("s1", "one", _bbox(15, 50, 35, 70)),
            target_field=None,
            match_type=MatchType.MISSING_IN_TARGET,
            confidence=1.0,
        ),
        MatchResult(
            source_field=_field("s2", "two", _bbox(37, 50, 57, 70)),
            target_field=None,
            match_type=MatchType.MISSING_IN_TARGET,
            confidence=1.0,
        ),
    ]
    artifacts = _compare_artifacts(results=results, source_image=source, target_image=source)
    manifest = build_visual_diff_manifest(_response(results), artifacts)
    placements = plan_badge_placements(manifest)
    highlight_rects = [
        (
            int(round(ann.overlay_bbox_normalized.x1 * 120)),
            int(round(ann.overlay_bbox_normalized.y1 * 120)),
            int(round(ann.overlay_bbox_normalized.x2 * 120)),
            int(round(ann.overlay_bbox_normalized.y2 * 120)),
        )
        for ann in manifest.annotations
    ]

    for p in placements:
        badge = (p.x1, p.y1, p.x2, p.y2)
        assert all(
            badge[2] <= rect[0]
            or rect[2] <= badge[0]
            or badge[3] <= rect[1]
            or rect[3] <= badge[1]
            for rect in highlight_rects
        )


def test_plan_badge_placements_falls_back_inside_for_edge_box() -> None:
    source = np.zeros((20, 20, 3), dtype=np.uint8)
    src = _field("s1", "edge", _bbox(0, 0, 4, 4))
    result = MatchResult(
        source_field=src,
        target_field=None,
        match_type=MatchType.MISSING_IN_TARGET,
        confidence=1.0,
    )
    artifacts = _compare_artifacts(results=[result], source_image=source, target_image=source)
    manifest = build_visual_diff_manifest(_response([result]), artifacts)

    placement = plan_badge_placements(manifest)[0]

    assert placement.placement == "inside"
    assert placement.x1 >= 0
    assert placement.y1 >= 0


def test_run_manual_compare_writes_minimal_reviewer_bundle() -> None:
    tmp_path = Path("data/tmp/test_visual_diff") / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        source = np.zeros((20, 20, 3), dtype=np.uint8)
        ok_src, enc_src = cv2.imencode(".png", source)
        ok_tgt, enc_tgt = cv2.imencode(".png", source)
        assert ok_src and ok_tgt
        source_path = tmp_path / "source.png"
        target_path = tmp_path / "target.png"
        source_path.write_bytes(bytes(enc_src.tobytes()))
        target_path.write_bytes(bytes(enc_tgt.tobytes()))

        outputs = run_manual_compare(
            source=source_path,
            target=target_path,
            ocr_provider="stub",
            out_root=tmp_path,
            run_id="smoke_visual",
        )

        assert outputs["raw_json"] is None
        assert outputs["report_json"] is None
        assert outputs["report_md"] is None
        assert outputs["visual_manifest"] is None
        assert outputs["visual_overlay_png"] is None
        assert outputs["visual_report"] is None
        assert Path(outputs["llm_usage_json"]).is_file()
        assert Path(outputs["visual_overlay"]).is_file()
        assert Path(outputs["comparison_docx"]).is_file()
        assert Path(outputs["summary_json"]).is_file()

        out_dir = Path(outputs["out_dir"])
        assert {path.name for path in out_dir.iterdir() if path.is_file()} == {
            "comparison_llm_usage.json",
            "comparison_summary.docx",
            "visual_diff_overlay.pdf",
            "run_summary.json",
        }

        summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
        assert summary["primary_reviewer_artifacts"] == {
            "visual_overlay": "visual_diff_overlay.pdf",
            "visual_overlay_kind": "pdf",
            "comparison_summary_docx": "comparison_summary.docx",
            "comparison_llm_usage_json": "comparison_llm_usage.json",
            "summary_json": "run_summary.json",
        }
        assert summary["visual_overlay_kind"] == "pdf"
        assert isinstance(summary["comparison_llm_usage"], dict)
        assert summary["debug_artifacts"] is None
        assert Path(outputs["visual_overlay"]).read_bytes().startswith(b"%PDF-")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_manual_compare_full_artifacts_writes_debug_bundle() -> None:
    tmp_path = Path("data/tmp/test_visual_diff") / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        source = np.zeros((20, 20, 3), dtype=np.uint8)
        ok_src, enc_src = cv2.imencode(".png", source)
        ok_tgt, enc_tgt = cv2.imencode(".png", source)
        assert ok_src and ok_tgt
        source_path = tmp_path / "source.png"
        target_path = tmp_path / "target.png"
        source_path.write_bytes(bytes(enc_src.tobytes()))
        target_path.write_bytes(bytes(enc_tgt.tobytes()))

        outputs = run_manual_compare(
            source=source_path,
            target=target_path,
            ocr_provider="stub",
            out_root=tmp_path,
            run_id="smoke_visual_full",
            full_artifacts=True,
        )

        for key in (
            "raw_json",
            "report_json",
            "report_md",
            "comparison_docx",
            "visual_manifest",
            "visual_overlay",
            "visual_overlay_png",
            "visual_report",
            "summary_json",
        ):
            assert outputs[key] is not None
            assert Path(outputs[key]).is_file()

        out_dir = Path(outputs["out_dir"])
        assert {path.name for path in out_dir.iterdir() if path.is_file()} == {
            "compare_response.json",
            "comparison_report.json",
            "comparison_report.md",
            "comparison_llm_usage.json",
            "comparison_summary.docx",
            "run_summary.json",
            "visual_diff_manifest.json",
            "visual_diff_overlay.pdf",
            "visual_diff_overlay.png",
            "visual_diff_report.html",
        }
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
