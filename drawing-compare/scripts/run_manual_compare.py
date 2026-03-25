#!/usr/bin/env python3
"""Run a one-off drawing compare and persist response artifacts to disk."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.api.deps import build_compare_service
from app.reporting.markdown_report import build_comparison_markdown
from app.reporting.reviewer_bundle import build_reviewer_bundle
from app.reporting.visual_diff import image_to_png_bytes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compare on two local files and write the reviewer bundle "
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
    parser.add_argument(
        "--full-artifacts",
        action="store_true",
        help="Also write the legacy debug bundle (raw JSON, manifest, HTML, and PNG overlay).",
    )
    return parser.parse_args()


def run_manual_compare(
    *,
    source: Path,
    target: Path,
    ocr_provider: str,
    out_root: Path,
    run_id: str | None = None,
    full_artifacts: bool = False,
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
    bundle = build_reviewer_bundle(
        resp,
        compare_artifacts,
        include_debug_artifacts=full_artifacts,
    )

    raw_json_path = out_dir / "compare_response.json"
    report_json_path = out_dir / "comparison_report.json"
    report_md_path = out_dir / "comparison_report.md"
    visual_manifest_path = out_dir / "visual_diff_manifest.json"
    visual_overlay_path = out_dir / bundle.visual_filename
    comparison_docx_path = out_dir / "comparison_summary.docx"
    visual_overlay_png_path = out_dir / "visual_diff_overlay.png"
    visual_report_path = out_dir / "visual_diff_report.html"
    summary_path = out_dir / "run_summary.json"

    visual_overlay_path.write_bytes(bundle.visual_bytes)
    comparison_docx_path.write_bytes(bundle.comparison_docx)

    debug_artifacts: dict[str, str] = {}
    if full_artifacts:
        report_md = build_comparison_markdown(
            resp,
            annotated_findings=bundle.manifest.annotations,
            annotated_pdf_href=visual_overlay_path.name,
            annotated_visual_kind=bundle.visual_kind,
        )
        raw_json_path.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        report_json = str(resp.extras.get("comparison_json_report", ""))
        report_json_path.write_text(report_json, encoding="utf-8")
        report_md_path.write_text(report_md, encoding="utf-8")
        visual_manifest_path.write_text(bundle.manifest.model_dump_json(indent=2), encoding="utf-8")
        if bundle.visual_kind != "png":
            visual_overlay_png_path.write_bytes(image_to_png_bytes(bundle.overlay_bgr))
        elif visual_overlay_path != visual_overlay_png_path:
            visual_overlay_png_path.write_bytes(bundle.visual_bytes)
        if bundle.html is not None:
            visual_report_path.write_text(bundle.html, encoding="utf-8")
        debug_artifacts = {
            "raw_json": raw_json_path.name,
            "report_json": report_json_path.name,
            "report_md": report_md_path.name,
            "visual_manifest": visual_manifest_path.name,
            "visual_overlay_png": visual_overlay_png_path.name,
            "visual_report": visual_report_path.name,
        }

    primary_artifacts = {
        "visual_overlay": visual_overlay_path.name,
        "visual_overlay_kind": bundle.visual_kind,
        "comparison_summary_docx": comparison_docx_path.name,
        "summary_json": summary_path.name,
    }
    summary_path.write_text(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "ocr_provider": ocr_provider,
                "source_path": str(source),
                "target_path": str(target),
                "summary": resp.report.summary.model_dump(),
                "review_flags": len(resp.review_flags),
                "visual_annotation_count": bundle.manifest.annotation_count,
                "visual_overlay_kind": bundle.visual_kind,
                "full_artifacts": full_artifacts,
                "primary_reviewer_artifacts": primary_artifacts,
                "debug_artifacts": debug_artifacts or None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "run_id": resolved_run_id,
        "out_dir": out_dir.resolve(),
        "raw_json": raw_json_path.resolve() if full_artifacts else None,
        "report_json": report_json_path.resolve() if full_artifacts else None,
        "report_md": report_md_path.resolve() if full_artifacts else None,
        "comparison_docx": comparison_docx_path.resolve(),
        "visual_manifest": visual_manifest_path.resolve() if full_artifacts else None,
        "visual_overlay": visual_overlay_path.resolve(),
        "visual_overlay_png": visual_overlay_png_path.resolve() if full_artifacts else None,
        "visual_report": visual_report_path.resolve() if full_artifacts else None,
        "summary_json": summary_path.resolve(),
        "visual_annotation_count": bundle.manifest.annotation_count,
        "visual_overlay_kind": bundle.visual_kind,
        "full_artifacts": full_artifacts,
    }


def main() -> int:
    args = _parse_args()
    outputs = run_manual_compare(
        source=args.source,
        target=args.target,
        ocr_provider=args.ocr_provider,
        out_root=args.out_root,
        run_id=args.run_id,
        full_artifacts=args.full_artifacts,
    )

    print(f"OUT_DIR={outputs['out_dir']}")
    print(f"VISUAL_OVERLAY={outputs['visual_overlay']}")
    print(f"COMPARISON_DOCX={outputs['comparison_docx']}")
    print(f"SUMMARY_JSON={outputs['summary_json']}")
    print(f"VISUAL_OVERLAY_KIND={outputs['visual_overlay_kind']}")
    if outputs["report_md"] is not None:
        print(f"REPORT_MD={outputs['report_md']}")
    if outputs["raw_json"] is not None:
        print(f"RAW_JSON={outputs['raw_json']}")
    if outputs["report_json"] is not None:
        print(f"REPORT_JSON={outputs['report_json']}")
    if outputs["visual_manifest"] is not None:
        print(f"VISUAL_MANIFEST={outputs['visual_manifest']}")
    if outputs["visual_overlay_png"] is not None:
        print(f"VISUAL_OVERLAY_PNG={outputs['visual_overlay_png']}")
    if outputs["visual_report"] is not None:
        print(f"VISUAL_REPORT={outputs['visual_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
