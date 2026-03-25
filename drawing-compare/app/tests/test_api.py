"""API smoke tests."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app


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
    (older / "text_diff_report.html").write_text("<html><body>old</body></html>", encoding="utf-8")
    (newer / "text_diff_report.html").write_text("<html><body>newest</body></html>", encoding="utf-8")

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
