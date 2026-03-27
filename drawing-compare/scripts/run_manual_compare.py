#!/usr/bin/env python3
"""Run a one-off drawing compare and persist response artifacts to disk."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from app.api.deps import build_compare_service
from app.core.config import Settings, get_settings
from app.ocr.base import OCRResult
from app.ocr.factory import get_local_ocr_provider, get_ocr_result_provider
from app.ocr.normalizer import normalize_ocr_result
from app.reporting.markdown_report import build_comparison_markdown
from app.reporting.reviewer_bundle import build_reviewer_bundle
from app.reporting.visual_diff import image_to_png_bytes


def _effective_settings_for_run(
    *,
    ocr_provider: str,
    ocr_result_backend: str | None,
    dual_ocr: bool,
) -> Settings:
    base = get_settings()
    updates: dict[str, Any] = {}
    if ocr_provider and ocr_provider.strip():
        updates["ocr_provider"] = ocr_provider.strip().lower()
    if ocr_result_backend is not None:
        updates["ocr_result_backend"] = ocr_result_backend
    if dual_ocr:
        updates["enable_dual_ocr"] = True
    return base.model_copy(update=updates) if updates else base


def _merge_ocr_results(left: OCRResult, right: OCRResult) -> OCRResult:
    n = len(left.pages)
    merged_pages = list(left.pages)
    for i, p in enumerate(right.pages):
        merged_pages.append(p.model_copy(update={"page_number": n + i + 1}))
    return OCRResult(pages=merged_pages)


def _try_raw_extract(fn: Callable[[], OCRResult]) -> dict[str, Any]:
    try:
        r = fn()
        return {"ok": True, "ocr": r.model_dump(mode="json")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _normalize_entry(raw_entry: dict[str, Any]) -> dict[str, Any]:
    if not raw_entry.get("ok"):
        return {"ok": False, "error": raw_entry.get("error", "unknown")}
    try:
        raw = OCRResult.model_validate(raw_entry["ocr"])
        normalized = normalize_ocr_result(raw)
        return {"ok": True, "ocr": normalized.model_dump(mode="json")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _ocr_if_ok(entry: dict[str, Any]) -> OCRResult | None:
    if not entry.get("ok"):
        return None
    try:
        return OCRResult.model_validate(entry["ocr"])
    except Exception:
        return None


def _skip_google_ocr_experiment() -> bool:
    v = (os.environ.get("DRAWING_COMPARE_SKIP_GOOGLE_OCR_EXPERIMENT") or "").strip().lower()
    return v in ("1", "true", "yes")


def _google_raw_entry(google_prov: object, path: Path) -> dict[str, Any]:
    if _skip_google_ocr_experiment():
        return {
            "ok": False,
            "error": "skipped_DRAWING_COMPARE_SKIP_GOOGLE_OCR_EXPERIMENT",
        }
    return _try_raw_extract(lambda: google_prov.extract_raw(str(path)))


def _write_ocr_experiment_artifacts(
    out_dir: Path,
    source: Path,
    target: Path,
    *,
    ocr_provider: str,
    ocr_result_backend: str | None,
    dual_ocr: bool,
) -> dict[str, str]:
    """Write per-run OCR payloads under ``out_dir/ocr/``. Returns filename → relative path."""
    eff = _effective_settings_for_run(
        ocr_provider=ocr_provider,
        ocr_result_backend=ocr_result_backend,
        dual_ocr=dual_ocr,
    )
    local_settings = eff.model_copy(update={"ocr_result_backend": "local"})
    google_settings = eff.model_copy(update={"ocr_result_backend": "google"})
    local_prov = get_local_ocr_provider(local_settings)
    google_prov = get_ocr_result_provider(google_settings)

    ocr_dir = out_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    local_raw: dict[str, Any] = {
        "source": _try_raw_extract(lambda: local_prov.extract_raw(str(source))),
        "target": _try_raw_extract(lambda: local_prov.extract_raw(str(target))),
    }
    google_raw: dict[str, Any] = {
        "source": _google_raw_entry(google_prov, source),
        "target": _google_raw_entry(google_prov, target),
    }
    normalized: dict[str, Any] = {
        "source": {
            "local": _normalize_entry(local_raw["source"]),
            "google": _normalize_entry(google_raw["source"]),
        },
        "target": {
            "local": _normalize_entry(local_raw["target"]),
            "google": _normalize_entry(google_raw["target"]),
        },
    }

    metrics_payload: dict[str, Any]
    try:
        from app.ocr.ocr_comparator import compare_ocr

        ml = _ocr_if_ok(normalized["source"]["local"])
        mt = _ocr_if_ok(normalized["target"]["local"])
        mg_src = _ocr_if_ok(normalized["source"]["google"])
        mg_tgt = _ocr_if_ok(normalized["target"]["google"])
        if ml is not None and mt is not None and mg_src is not None and mg_tgt is not None:
            merged_local = _merge_ocr_results(ml, mt)
            merged_google = _merge_ocr_results(mg_src, mg_tgt)
            m = compare_ocr(merged_local, merged_google)
            metrics_payload = {
                "ok": True,
                **m.model_dump(mode="json"),
                "summary": m.summary(),
            }
        else:
            metrics_payload = {
                "ok": False,
                "reason": "incomplete_normalized_ocr_for_merged_compare",
            }
    except Exception as exc:
        metrics_payload = {"ok": False, "error": str(exc)}

    def _write(name: str, payload: Any) -> str:
        p = ocr_dir / name
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p.relative_to(out_dir).as_posix()

    return {
        "local_raw": _write("local_raw.json", local_raw),
        "google_raw": _write("google_raw.json", google_raw),
        "normalized": _write("normalized.json", normalized),
        "ocr_comparison_metrics": _write("ocr_comparison_metrics.json", metrics_payload),
    }


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
        "--ocr-backend",
        choices=["local", "google"],
        default=None,
        dest="ocr_result_backend",
        help=(
            "Normalized OCRResult backend for PDF document OCR. "
            "Maps to Settings.ocr_result_backend. Omit to use config default (typically local)."
        ),
    )
    parser.add_argument(
        "--ocr-engine",
        default="echo",
        help=(
            "Engine OCR for ndarray / bridge: stub, echo, stub_document, tesseract, "
            "paddleocr, windows_ocr/windows. Maps to Settings.ocr_provider. Default: echo."
        ),
    )
    parser.add_argument(
        "--dual-ocr",
        action="store_true",
        help="Run local and Google OCR pipelines independently (Settings.enable_dual_ocr).",
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
    ocr_result_backend: str | None = None,
    dual_ocr: bool = False,
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
        ocr_result_backend=ocr_result_backend,
        enable_dual_ocr=True if dual_ocr else None,
    )
    ocr_artifact_paths = _write_ocr_experiment_artifacts(
        out_dir,
        source,
        target,
        ocr_provider=ocr_provider,
        ocr_result_backend=ocr_result_backend,
        dual_ocr=dual_ocr,
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
    llm_usage_path = out_dir / "comparison_llm_usage.json"
    summary_path = out_dir / "run_summary.json"

    visual_overlay_path.write_bytes(bundle.visual_bytes)
    comparison_docx_path.write_bytes(bundle.comparison_docx)
    llm_usage = resp.extras.get("comparison_llm_usage")
    llm_usage_payload = llm_usage if isinstance(llm_usage, dict) else {}
    llm_usage_path.write_text(json.dumps(llm_usage_payload, indent=2), encoding="utf-8")

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
        "comparison_llm_usage_json": llm_usage_path.name,
        "summary_json": summary_path.name,
    }
    summary_path.write_text(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "ocr_provider": ocr_provider,
                "ocr_result_backend": ocr_result_backend,
                "dual_ocr": dual_ocr,
                "source_path": str(source),
                "target_path": str(target),
                "summary": resp.report.summary.model_dump(),
                "review_flags": len(resp.review_flags),
                "visual_annotation_count": bundle.manifest.annotation_count,
                "visual_overlay_kind": bundle.visual_kind,
                "full_artifacts": full_artifacts,
                "primary_reviewer_artifacts": primary_artifacts,
                "comparison_llm_usage": llm_usage_payload,
                "debug_artifacts": debug_artifacts or None,
                "extras_mode": resp.extras.get("mode"),
                "ocr_artifacts": ocr_artifact_paths,
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
        "llm_usage_json": llm_usage_path.resolve(),
        "visual_manifest": visual_manifest_path.resolve() if full_artifacts else None,
        "visual_overlay": visual_overlay_path.resolve(),
        "visual_overlay_png": visual_overlay_png_path.resolve() if full_artifacts else None,
        "visual_report": visual_report_path.resolve() if full_artifacts else None,
        "summary_json": summary_path.resolve(),
        "visual_annotation_count": bundle.manifest.annotation_count,
        "visual_overlay_kind": bundle.visual_kind,
        "full_artifacts": full_artifacts,
        "response_extras": dict(resp.extras),
        "ocr_artifacts": ocr_artifact_paths,
    }


def main() -> int:
    args = _parse_args()
    outputs = run_manual_compare(
        source=args.source,
        target=args.target,
        ocr_provider=args.ocr_engine,
        out_root=args.out_root,
        run_id=args.run_id,
        full_artifacts=args.full_artifacts,
        ocr_result_backend=args.ocr_result_backend,
        dual_ocr=args.dual_ocr,
    )

    print(f"OUT_DIR={outputs['out_dir']}")
    print(f"VISUAL_OVERLAY={outputs['visual_overlay']}")
    print(f"COMPARISON_DOCX={outputs['comparison_docx']}")
    print(f"SUMMARY_JSON={outputs['summary_json']}")
    print(f"OCR_DIR={Path(outputs['out_dir']) / 'ocr'}")
    print(f"OCR_ARTIFACTS={json.dumps(outputs.get('ocr_artifacts') or {}, sort_keys=True)}")
    print(f"VISUAL_OVERLAY_KIND={outputs['visual_overlay_kind']}")
    extras = outputs.get("response_extras") or {}
    if extras.get("mode") == "dual_ocr":
        print("=== DUAL OCR: local branch ===")
        print(json.dumps(extras.get("local"), indent=2, default=str))
        print("=== DUAL OCR: google branch ===")
        print(json.dumps(extras.get("google"), indent=2, default=str))
        print("=== DUAL OCR: ocr_comparison ===")
        print(json.dumps(extras.get("ocr_comparison"), indent=2, default=str))
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
