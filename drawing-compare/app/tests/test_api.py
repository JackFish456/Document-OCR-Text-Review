"""API smoke tests."""

import os
from pathlib import Path
from unittest.mock import patch

import cv2
import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import reset_settings_cache
from app.main import create_app
from app.services.compare import DrawingCompareService


def test_latest_text_diff_no_runs_returns_instructions(tmp_path: Path) -> None:
    """Empty manual_runs (isolated repo root) => fallback HTML with run instructions."""
    root = tmp_path
    (root / "experiments" / "manual_runs").mkdir(parents=True)
    with patch("app.api.routes.reports._repo_root", return_value=root):
        client = TestClient(create_app())
        r = client.get("/reports/latest-text-diff")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "No text diff report found" in r.text
    assert "compare_pdf_text.py" in r.text


def test_latest_text_diff_serves_newest_html(tmp_path: Path) -> None:
    manual = tmp_path / "experiments" / "manual_runs"
    older = manual / "textdiff_20000101_000000"
    newer = manual / "textdiff_20000102_000000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    old_path = older / "text_diff_report.html"
    new_path = newer / "text_diff_report.html"
    old_path.write_text("<html><body>old</body></html>", encoding="utf-8")
    new_path.write_text("<html><body>newest</body></html>", encoding="utf-8")
    # Deterministic ordering: selection uses mtime; Windows can stamp both writes in the same tick.
    os.utime(old_path, (1_000_000, 1_000_000))
    os.utime(new_path, (2_000_000, 2_000_000))

    with patch("app.api.routes.reports._repo_root", return_value=tmp_path):
        client = TestClient(create_app())
        r = client.get("/reports/latest-text-diff")

    assert r.status_code == 200
    assert "newest" in r.text
    assert "old" not in r.text


def test_health_ok() -> None:
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "drawing-compare"
    assert "version" in body


def test_compare_paths_stub_response() -> None:
    client = TestClient(create_app())
    r = client.post("/compare/paths", json={})
    assert r.status_code == 200
    body = r.json()
    assert "comparison_id" in body
    assert body["report"]["summary"]["total_source"] == 0
    assert body["report"]["matches"] == []
    assert body["extras"]["mode"] == "stub"
    assert "comparison_json_report" in body["extras"]
    assert "comparison_markdown_report" in body["extras"]
    assert "Drawing comparison report" in body["extras"]["comparison_markdown_report"]


def _mini_png() -> bytes:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_compare_multipart_pipeline() -> None:
    client = TestClient(create_app())
    png = _mini_png()
    r = client.post(
        "/compare",
        files=[
            ("source_file", ("source.png", png, "image/png")),
            ("target_file", ("target.png", png, "image/png")),
        ],
        data={"ocr_provider": "stub"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extras"]["mode"] == "full"
    assert body["extras"]["ocr_provider"] == "stub"
    assert "report" in body
    assert "review_flags" in body
    assert "artifact_job_id" in body["extras"]
    assert body["extras"]["output_visual_kind"] == "pdf"
    assert body["extras"]["output_visual_href"].startswith("/compare/artifacts/")
    assert body["extras"]["output_pdf_href"] == body["extras"]["output_visual_href"]
    assert body["extras"]["output_word_href"].endswith("comparison_summary.docx")
    assert body["extras"]["output_llm_usage_href"].endswith("comparison_llm_usage.json")
    assert "output_html_href" not in body["extras"]
    assert "output_manifest_href" not in body["extras"]


def test_compare_multipart_artifact_pdf_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted visual PDF is served from GET /compare/artifacts/{job_id}/..."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("DRAWING_COMPARE_DATA_DIR", str(data_dir))
    reset_settings_cache()
    try:
        client = TestClient(create_app())
        png = _mini_png()
        r = client.post(
            "/compare",
            files=[
                ("source_file", ("source.png", png, "image/png")),
                ("target_file", ("target.png", png, "image/png")),
            ],
            data={"ocr_provider": "stub", "include_text_reports": "false"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "comparison_markdown_report" not in body["extras"]
        assert "comparison_json_report" not in body["extras"]
        assert body["extras"]["output_visual_kind"] == "pdf"
        pdf_href = body["extras"]["output_visual_href"]
        assert pdf_href.startswith("/compare/artifacts/")
        dl = client.get(pdf_href)
        assert dl.status_code == 200
        assert dl.content[:4] == b"%PDF"
        assert "application/pdf" in dl.headers.get("content-type", "")
        word_href = body["extras"]["output_word_href"]
        assert word_href.endswith("comparison_summary.docx")
        wdl = client.get(word_href)
        assert wdl.status_code == 200
        assert wdl.content[:2] == b"PK"
        ct = wdl.headers.get("content-type", "")
        assert "wordprocessingml" in ct or "octet-stream" in ct
        usage_href = body["extras"]["output_llm_usage_href"]
        assert usage_href.endswith("comparison_llm_usage.json")
        udl = client.get(usage_href)
        assert udl.status_code == 200
        assert "application/json" in udl.headers.get("content-type", "")
        usage_json = udl.json()
        assert isinstance(usage_json, dict)
    finally:
        reset_settings_cache()


def test_compare_paths_local_files_return_reviewer_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    source.write_bytes(_mini_png())
    target.write_bytes(_mini_png())
    monkeypatch.setenv("DRAWING_COMPARE_DATA_DIR", str(data_dir))
    reset_settings_cache()
    try:
        client = TestClient(create_app())
        r = client.post(
            "/compare/paths",
            json={
                "drawing_a_uri": str(source),
                "drawing_b_uri": str(target),
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        extras = body["extras"]
        assert extras["mode"] == "full"
        assert extras["output_visual_kind"] == "pdf"
        assert extras["output_visual_href"].endswith("visual_diff_overlay.pdf")
        assert extras["output_word_href"].endswith("comparison_summary.docx")
        assert extras["output_llm_usage_href"].endswith("comparison_llm_usage.json")
    finally:
        reset_settings_cache()


def test_compare_multipart_falls_back_to_png_when_pdf_build_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("DRAWING_COMPARE_DATA_DIR", str(data_dir))
    reset_settings_cache()
    try:
        client = TestClient(create_app())
        png = _mini_png()
        with patch(
            "app.reporting.reviewer_bundle.render_visual_diff_overlay_pdf_pages",
            side_effect=RuntimeError("pdf broke"),
        ):
            r = client.post(
                "/compare",
                files=[
                    ("source_file", ("source.png", png, "image/png")),
                    ("target_file", ("target.png", png, "image/png")),
                ],
                data={"ocr_provider": "stub", "include_text_reports": "false"},
            )
        assert r.status_code == 200, r.text
        extras = r.json()["extras"]
        assert extras["output_visual_kind"] == "png"
        assert extras["output_visual_href"].endswith("visual_diff_overlay.png")
        assert extras["output_png_href"] == extras["output_visual_href"]
        assert "output_pdf_href" not in extras
        visual = client.get(extras["output_visual_href"])
        assert visual.status_code == 200
        assert visual.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert "image/png" in visual.headers.get("content-type", "")
    finally:
        reset_settings_cache()


def test_static_app_uses_visual_output_contract() -> None:
    client = TestClient(create_app())
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "output_visual_href" in r.text
    assert "output_visual_kind" in r.text
    assert "output_html_href" not in r.text


def test_compare_multipart_pipeline_stub_document_provider() -> None:
    client = TestClient(create_app())
    png = _mini_png()
    r = client.post(
        "/compare",
        files=[
            ("source_file", ("source.png", png, "image/png")),
            ("target_file", ("target.png", png, "image/png")),
        ],
        data={"ocr_provider": "stub_document"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extras"]["mode"] == "full"
    assert body["extras"]["ocr_provider"] == "stub_document"


def test_compare_multipart_bad_extension() -> None:
    client = TestClient(create_app())
    r = client.post(
        "/compare",
        files=[
            ("source_file", ("source.bin", b"x", "application/octet-stream")),
            ("target_file", ("target.png", _mini_png(), "image/png")),
        ],
    )
    assert r.status_code == 400


def test_compare_batch_paths_label_mismatch_is_422(tmp_path: Path) -> None:
    baseline = tmp_path / "base.png"
    c1 = tmp_path / "c1.png"
    baseline.write_bytes(_mini_png())
    c1.write_bytes(_mini_png())
    client = TestClient(create_app())
    r = client.post(
        "/compare/paths/batch",
        json={
            "baseline_uri": str(baseline),
            "candidate_uris": [str(c1)],
            "candidate_labels": ["a", "extra"],
        },
    )
    assert r.status_code == 422


def test_compare_batch_paths_merged_pdf_two_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    baseline = tmp_path / "base.png"
    c1 = tmp_path / "c1.png"
    c2 = tmp_path / "c2.png"
    baseline.write_bytes(_mini_png())
    c1.write_bytes(_mini_png())
    c2.write_bytes(_mini_png())
    monkeypatch.setenv("DRAWING_COMPARE_DATA_DIR", str(data_dir))
    reset_settings_cache()
    try:
        client = TestClient(create_app())

        r_single = client.post(
            "/compare/paths",
            json={"drawing_a_uri": str(baseline), "drawing_b_uri": str(c1)},
        )
        assert r_single.status_code == 200, r_single.text
        href_single = r_single.json()["extras"]["output_visual_href"]
        pdf_single = client.get(href_single).content
        doc_single = fitz.open(stream=pdf_single, filetype="pdf")
        try:
            n_pages_one = doc_single.page_count
        finally:
            doc_single.close()
        assert n_pages_one >= 1

        r = client.post(
            "/compare/paths/batch",
            json={
                "baseline_uri": str(baseline),
                "candidate_uris": [str(c1), str(c2)],
                "candidate_labels": ["first", "second"],
                "include_text_reports": False,
                "ocr_provider": "stub",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["pair_count"] == 2
        assert body["pairs_succeeded"] == 2
        assert len(body["pairs"]) == 2
        assert body["pairs"][0]["ok"] is True
        assert body["pairs"][0]["candidate_label"] == "first"
        assert body["pairs"][1]["ok"] is True
        assert body["pairs"][1]["candidate_label"] == "second"
        assert "comparison_json_report" not in body["pairs"][0]["pair_extras"]
        extras = body["extras"]
        assert extras["batch_mode"] == "baseline_many"
        assert extras["output_merged_pdf_href"] == extras["output_pdf_href"]
        assert extras["output_visual_href"].endswith("visual_diff_overlay_merged.pdf")
        merged = client.get(extras["output_merged_pdf_href"]).content
        assert merged[:4] == b"%PDF"
        doc_merged = fitz.open(stream=merged, filetype="pdf")
        try:
            assert doc_merged.page_count == n_pages_one * 2
        finally:
            doc_merged.close()
    finally:
        reset_settings_cache()


def test_compare_batch_paths_partial_failure_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    baseline = tmp_path / "base.png"
    c1 = tmp_path / "c1.png"
    c2 = tmp_path / "c2.png"
    baseline.write_bytes(_mini_png())
    c1.write_bytes(_mini_png())
    c2.write_bytes(_mini_png())
    monkeypatch.setenv("DRAWING_COMPARE_DATA_DIR", str(data_dir))
    reset_settings_cache()
    try:
        client = TestClient(create_app())
        orig = DrawingCompareService.compare_paths_with_artifacts
        call_count = {"n": 0}

        def wrapped(self: DrawingCompareService, *a: object, **kw: object):
            call_count["n"] += 1
            if call_count["n"] == 2:
                msg = "simulated pair failure"
                raise ValueError(msg)
            return orig(self, *a, **kw)

        with patch.object(DrawingCompareService, "compare_paths_with_artifacts", wrapped):
            r = client.post(
                "/compare/paths/batch",
                json={
                    "baseline_uri": str(baseline),
                    "candidate_uris": [str(c1), str(c2)],
                    "fail_fast": False,
                    "ocr_provider": "stub",
                    "include_text_reports": False,
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["pairs_succeeded"] == 1
        assert body["pairs"][0]["ok"] is True
        assert body["pairs"][1]["ok"] is False
        assert body["pairs"][1]["error"] is not None
        extras = body["extras"]
        merged = client.get(extras["output_merged_pdf_href"]).content
        doc_merged = fitz.open(stream=merged, filetype="pdf")
        try:
            r0 = client.post(
                "/compare/paths",
                json={"drawing_a_uri": str(baseline), "drawing_b_uri": str(c1)},
            )
            p0 = client.get(r0.json()["extras"]["output_visual_href"]).content
            d0 = fitz.open(stream=p0, filetype="pdf")
            try:
                assert doc_merged.page_count == d0.page_count
            finally:
                d0.close()
        finally:
            doc_merged.close()
    finally:
        reset_settings_cache()


def test_compare_batch_paths_fail_fast_returns_400(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    baseline = tmp_path / "base.png"
    c1 = tmp_path / "c1.png"
    c2 = tmp_path / "c2.png"
    baseline.write_bytes(_mini_png())
    c1.write_bytes(_mini_png())
    c2.write_bytes(_mini_png())
    monkeypatch.setenv("DRAWING_COMPARE_DATA_DIR", str(data_dir))
    reset_settings_cache()
    try:
        client = TestClient(create_app())
        orig = DrawingCompareService.compare_paths_with_artifacts
        call_count = {"n": 0}

        def wrapped(self: DrawingCompareService, *a: object, **kw: object):
            call_count["n"] += 1
            if call_count["n"] == 2:
                msg = "simulated pair failure"
                raise ValueError(msg)
            return orig(self, *a, **kw)

        with patch.object(DrawingCompareService, "compare_paths_with_artifacts", wrapped):
            r = client.post(
                "/compare/paths/batch",
                json={
                    "baseline_uri": str(baseline),
                    "candidate_uris": [str(c1), str(c2)],
                    "fail_fast": True,
                    "ocr_provider": "stub",
                },
            )
        assert r.status_code == 400
        assert "simulated pair failure" in (r.json().get("detail") or "")
    finally:
        reset_settings_cache()
