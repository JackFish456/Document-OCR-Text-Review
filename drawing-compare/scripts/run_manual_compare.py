#!/usr/bin/env python3
"""Run a one-off drawing compare and persist response artifacts to disk."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.api.deps import build_compare_service
from app.reporting.visual_diff import build_visual_diff_artifacts, image_to_png_bytes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compare on two local files and write JSON + Markdown artifacts "
            "under experiments/manual_runs/<run-id>."
        )
    )
    parser.add_argument("--source", required=True, type=Path, help="Source drawing path")
    parser.add_argument("--target", required=True, type=Path, help="Target drawing path")
    parser.add_argument(
        "--ocr-provider",
        default="echo",
        help="OCR provider (stub, echo, stub_document, tesseract, paddleocr, windows_ocr/windows)",
    )
    parser.add_argument(
        "--out-root",
        default=Path("experiments/manual_runs"),
        type=Path,
        help="Output root directory",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id")
    return parser.parse_args()


def run_manual_compare(
    *,
    source: Path,
    target: Path,
    ocr_provider: str,
    out_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source file: {source}")
    if not target.is_file():
        raise FileNotFoundError(f"Missing target file: {target}")

    resolved_run_id = run_id or datetime.now().strftime("manual_%Y%m%d_%H%M%S")
    out_dir = out_root / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    svc = build_compare_service()
    resp, compare_artifacts = svc.compare_paths_with_artifacts(
        source,
        target,
        ocr_provider=ocr_provider,
    )
    visual = build_visual_diff_artifacts(resp, compare_artifacts)

    raw_json_path = out_dir / "compare_response.json"
    report_json_path = out_dir / "comparison_report.json"
    report_md_path = out_dir / "comparison_report.md"
    visual_manifest_path = out_dir / "visual_diff_manifest.json"
    visual_overlay_path = out_dir / "visual_diff_overlay.png"
    visual_report_path = out_dir / "visual_diff_report.html"
    summary_path = out_dir / "run_summary.json"

    raw_json_path.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
    report_json = str(resp.extras.get("comparison_json_report", ""))
    report_md = str(resp.extras.get("comparison_markdown_report", ""))
    report_json_path.write_text(report_json, encoding="utf-8")
    report_md_path.write_text(report_md, encoding="utf-8")
    visual_manifest_path.write_text(visual.manifest.model_dump_json(indent=2), encoding="utf-8")
    visual_overlay_path.write_bytes(image_to_png_bytes(visual.overlay_bgr))
    visual_report_path.write_text(visual.html, encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "ocr_provider": ocr_provider,
                "source_path": str(source),
                "target_path": str(target),
                "summary": resp.report.summary.model_dump(),
                "review_flags": len(resp.review_flags),
                "visual_annotation_count": visual.manifest.annotation_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "run_id": resolved_run_id,
        "out_dir": out_dir.resolve(),
        "raw_json": raw_json_path.resolve(),
        "report_json": report_json_path.resolve(),
        "report_md": report_md_path.resolve(),
        "visual_manifest": visual_manifest_path.resolve(),
        "visual_overlay": visual_overlay_path.resolve(),
        "visual_report": visual_report_path.resolve(),
        "summary_json": summary_path.resolve(),
        "visual_annotation_count": visual.manifest.annotation_count,
    }


def main() -> int:
    args = _parse_args()
    outputs = run_manual_compare(
        source=args.source,
        target=args.target,
        ocr_provider=args.ocr_provider,
        out_root=args.out_root,
        run_id=args.run_id,
    )

    print(f"OUT_DIR={outputs['out_dir']}")
    print(f"RAW_JSON={outputs['raw_json']}")
    print(f"REPORT_JSON={outputs['report_json']}")
    print(f"REPORT_MD={outputs['report_md']}")
    print(f"VISUAL_MANIFEST={outputs['visual_manifest']}")
    print(f"VISUAL_OVERLAY={outputs['visual_overlay']}")
    print(f"VISUAL_REPORT={outputs['visual_report']}")
    print(f"SUMMARY_JSON={outputs['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
