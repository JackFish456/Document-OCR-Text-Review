"""Visual diff manifest, renderer, and manual-run artifact tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.models.comparison import CompareResponse
from app.models.extraction import ExtractedField
from app.models.match import ComparisonReport, MatchResult, MatchType
from app.models.ocr import BoundingBox
from app.preprocessing.types import PreprocessPageResult
from app.reporting.visual_diff import (
    build_visual_diff_artifacts,
    build_visual_diff_manifest,
    render_visual_diff_overlay,
)
from app.services.compare_artifacts import CompareArtifacts
from scripts.run_manual_compare import run_manual_compare


def _bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _field(field_id: str, text: str, bbox: BoundingBox) -> ExtractedField:
    return ExtractedField(
        field_id=field_id,
        value=text,
        raw_text=text,
        bbox=bbox,
        confidence=0.95,
    )


def _page(image: np.ndarray, *, source_path: str = "memory.png") -> PreprocessPageResult:
    h, w = image.shape[:2]
    return PreprocessPageResult(
        page_number=1,
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
        source_page=_page(source_image, source_path="source.png"),
        target_page=_page(target_image, source_path="target.png"),
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
    assert changed.overlay_bbox_normalized.x1 == 0.25
    missing = manifest.annotations[1]
    assert missing.overlay_bbox_normalized.x1 == 0.2
    extra = manifest.annotations[2]
    assert extra.overlay_bbox_normalized.x1 == 0.55


def test_render_visual_diff_maps_target_overlay_into_source_space() -> None:
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

    assert rendered[30, 60].sum() > 0
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
    assert "Visual Overlay Diff Report" in bundle.html
    assert "data-toggle-type=\"changed_value\"" in bundle.html


def test_run_manual_compare_writes_visual_artifacts() -> None:
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

        for key in (
            "raw_json",
            "report_json",
            "report_md",
            "visual_manifest",
            "visual_overlay",
            "visual_report",
            "summary_json",
        ):
            assert Path(outputs[key]).is_file()

        manifest = json.loads(Path(outputs["visual_manifest"]).read_text(encoding="utf-8"))
        summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
        non_exact = (
            summary["summary"]["changed"]
            + summary["summary"]["missing"]
            + summary["summary"]["extra_target"]
            + summary["summary"]["uncertain"]
        )
        assert manifest["annotation_count"] == non_exact
        visual_report = Path(outputs["visual_report"]).read_text(encoding="utf-8")
        assert "Visual Overlay Diff Report" in visual_report
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
