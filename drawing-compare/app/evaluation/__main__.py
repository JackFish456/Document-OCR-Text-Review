"""CLI batch evaluation: python -m app.evaluation (from repo root)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the drawing-compare pipeline on each golden manifest pair and "
            "emit JSON, CSV, and a console report (precision, recall, F1, confusion matrix)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="example_manifest.json",
        help="Manifest filename under data/golden/manifests/",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root for resolving paths (default: parent of data/golden)",
    )
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=None,
        help="Golden dataset root (default: <project-root>/data/golden)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write eval_report.json and CSVs here (default: experiments/eval_runs/<run-id>)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Stable id for this run (default: eval-<uuid>)",
    )
    parser.add_argument(
        "--ocr-provider",
        type=str,
        default=None,
        help="Override OCR provider passed to the compare pipeline",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Skip printing the text report to stdout",
    )
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
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"eval-{ts}"

    from app.evaluation.reporting import print_batch_eval_console, write_batch_eval_csvs, write_batch_eval_json
    from app.evaluation.runner import run_batch_golden_evaluation

    report = run_batch_golden_evaluation(
        args.manifest,
        project_root=project_root,
        golden_root=golden_root,
        run_id=run_id,
        ocr_provider=args.ocr_provider,
    )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = project_root / "experiments" / "eval_runs" / report.run_id
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_batch_eval_json(report, out_dir / "eval_report.json")
    write_batch_eval_csvs(report, out_dir)

    if not args.no_console:
        print_batch_eval_console(report)
    else:
        print(f"Wrote JSON and CSVs to {out_dir}", file=sys.stderr)

    return 0 if report.pairs_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
