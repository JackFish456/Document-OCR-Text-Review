"""CLI: python -m app.datasets (run from repo root with package on PYTHONPATH)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate golden dataset manifests.")
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=None,
        help="Path to data/golden (default: <repo>/data/golden from this package location)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repo root for resolving relative drawing paths (default: parent of golden-root)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="example_manifest.json",
        help="Manifest filename under manifests/",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip existence checks for drawing and annotation paths",
    )
    args = parser.parse_args()

    golden_root = args.golden_root
    if golden_root is None:
        golden_root = Path(__file__).resolve().parent.parent.parent / "data" / "golden"

    project_root = args.project_root
    if project_root is None:
        project_root = golden_root.parent.parent

    from app.datasets.manifest import GoldenDatasetLoader
    from app.datasets.validator import validate_golden_dataset

    loader = GoldenDatasetLoader(golden_root)
    result = validate_golden_dataset(
        loader,
        project_root=project_root,
        manifest_name=args.manifest,
        check_files_exist=not args.no_check_files,
        load_annotations=True,
    )
    for w in result.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for err in result.errors:
        print(f"ERROR: {err}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
