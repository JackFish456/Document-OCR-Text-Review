"""Validate golden manifests, annotations, and on-disk paths."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.datasets.manifest import GoldenDatasetLoader, _parse_manifest_raw
from app.models.evaluation import (
    GoldenManifestIndex,
    GoldenPairAnnotation,
    GoldenPairManifestEntry,
)


@dataclass
class GoldenValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_manifest_entries(
    entries: list[GoldenPairManifestEntry],
    *,
    project_root: Path,
    check_files_exist: bool = True,
) -> GoldenValidationResult:
    """Check uniqueness and optional path existence for manifest rows."""
    out = GoldenValidationResult(ok=True)
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if e.pair_id in seen:
            out.add_error(f"Duplicate pair_id {e.pair_id!r} (row {i})")
        seen.add(e.pair_id)
        if check_files_exist:
            a = Path(e.drawing_a_ref)
            b = Path(e.drawing_b_ref)
            if not a.is_absolute():
                a = project_root / a
            if not b.is_absolute():
                b = project_root / b
            if not a.is_file():
                out.add_error(f"pair_id={e.pair_id}: drawing_a_ref not found: {a}")
            if not b.is_file():
                out.add_error(f"pair_id={e.pair_id}: drawing_b_ref not found: {b}")
            if e.annotation_path:
                ap = Path(e.annotation_path)
                if not ap.is_absolute():
                    ap = project_root / ap
                if not ap.is_file():
                    out.add_error(f"pair_id={e.pair_id}: annotation_path not found: {ap}")
    return out


def validate_annotation(ann: GoldenPairAnnotation) -> GoldenValidationResult:
    """Check annotation internal consistency."""
    out = GoldenValidationResult(ok=True)
    seen_ids: set[str] = set()
    for f in ann.fields:
        if f.field_id in seen_ids:
            out.add_error(f"Duplicate field_id in annotation: {f.field_id!r}")
        seen_ids.add(f.field_id)
    return out


def validate_annotation_matches_manifest(
    entry: GoldenPairManifestEntry,
    ann: GoldenPairAnnotation,
) -> GoldenValidationResult:
    out = GoldenValidationResult(ok=True)
    if ann.pair_id != entry.pair_id:
        out.add_error(
            f"pair_id mismatch: manifest has {entry.pair_id!r}, annotation has {ann.pair_id!r}"
        )
    return out


def validate_manifest_file(
    path: Path,
    *,
    project_root: Path,
    check_files_exist: bool = True,
) -> GoldenValidationResult:
    """Parse and validate one manifest JSON file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = _parse_manifest_raw(raw)
    except (OSError, ValueError) as ex:
        return GoldenValidationResult(ok=False, errors=[str(ex)])
    return validate_manifest_entries(
        entries, project_root=project_root, check_files_exist=check_files_exist
    )


def validate_index(index: GoldenManifestIndex, manifests_dir: Path) -> GoldenValidationResult:
    out = GoldenValidationResult(ok=True)
    for m in index.manifests:
        fp = manifests_dir / m.file
        if not fp.is_file():
            out.add_error(f"Index references missing manifest file: {m.file}")
    if index.default_manifest:
        df = manifests_dir / index.default_manifest
        if not df.is_file():
            out.add_warning(f"default_manifest not found: {index.default_manifest}")
    return out


def validate_golden_dataset(
    loader: GoldenDatasetLoader,
    *,
    project_root: Path,
    manifest_name: str,
    check_files_exist: bool = True,
    load_annotations: bool = True,
) -> GoldenValidationResult:
    """Validate one manifest and each referenced annotation (if present)."""
    out = GoldenValidationResult(ok=True)
    try:
        entries = loader.load_manifest_entries(manifest_name)
    except (OSError, ValueError) as ex:
        return GoldenValidationResult(ok=False, errors=[str(ex)])

    mres = validate_manifest_entries(
        entries, project_root=project_root, check_files_exist=check_files_exist
    )
    out.errors.extend(mres.errors)
    out.warnings.extend(mres.warnings)
    if not mres.ok:
        out.ok = False

    if not load_annotations:
        return out

    for e in entries:
        ann = loader.load_annotation_for_entry(e, project_root=project_root)
        if ann is None:
            out.add_warning(f"pair_id={e.pair_id}: no annotation file found")
            continue
        ares = validate_annotation(ann)
        out.errors.extend(ares.errors)
        out.warnings.extend(ares.warnings)
        if not ares.ok:
            out.ok = False
        m2 = validate_annotation_matches_manifest(e, ann)
        out.errors.extend(m2.errors)
        if not m2.ok:
            out.ok = False

    return out
