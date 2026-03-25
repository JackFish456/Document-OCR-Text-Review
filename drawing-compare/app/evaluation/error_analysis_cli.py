"""CLI: drawing-compare-error-analysis or python -m app.evaluation.error_analysis_cli"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize golden-set mismatches into OCR / extraction / matching / classification "
            "and write JSON + Markdown reports."
        ),
    )
    parser.add_argument("--manifest", type=str, default="example_manifest.json")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--golden-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--ocr-provider", type=str, default=None)
    parser.add_argument(
        "--ocr-confidence-threshold",
        type=float,
        default=0.55,
        help="Below this field confidence (paired rows), failures lean OCR when ambiguous",
    )
    parser.add_argument("--max-examples-per-category", type=int, default=8)
    parser.add_argument("--max-failures-json", type=int, default=500)
    parser.add_argument("--no-console", action="store_true")
    args = parser.parse_args()

    golden_root = args.golden_root
    if golden_root is None:
        golden_root = Path(__file__).resolve().parent.parent.parent / "data" / "golden"
    golden_root = golden_root.resolve()

    project_root = args.project_root
    if project_root is None:
        project_root = golden_root.parent.parent
    project_root = project_root.resolve()

    run_id = args.run_id
    if run_id is None:
        run_id = f"err-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    from app.evaluation.error_analysis_reporting import (
        print_error_analysis_console,
        write_error_analysis_json,
        write_error_analysis_markdown,
    )
    from app.evaluation.error_analysis_runner import run_batch_error_analysis

    report = run_batch_error_analysis(
        args.manifest,
        project_root=project_root,
        golden_root=golden_root,
        run_id=run_id,
        ocr_provider=args.ocr_provider,
        ocr_confidence_threshold=args.ocr_confidence_threshold,
        max_examples_per_category=args.max_examples_per_category,
    )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = project_root / "experiments" / "error_analysis" / report.run_id
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_error_analysis_json(report, out_dir / "error_analysis.json", max_failures_in_json=args.max_failures_json)
    write_error_analysis_markdown(report, out_dir / "error_analysis.md")

    if not args.no_console:
        print_error_analysis_console(report)
    else:
        print(f"Wrote {out_dir / 'error_analysis.json'} and error_analysis.md", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
