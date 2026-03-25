"""Load golden pair manifests, index, and per-pair annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app.models.evaluation import (
    GoldenManifestDocument,
    GoldenManifestIndex,
    GoldenPairAnnotation,
    GoldenPairManifestEntry,
)


@dataclass(frozen=True, slots=True)
class ResolvedGoldenPair:
    """Manifest entry with paths resolved against a project root."""

    pair_id: str
    drawing_a: Path
    drawing_b: Path
    annotation: Path | None
    entry: GoldenPairManifestEntry


def _parse_manifest_raw(raw: Any) -> list[GoldenPairManifestEntry]:
    if isinstance(raw, list):
        return TypeAdapter(list[GoldenPairManifestEntry]).validate_python(raw)
    if isinstance(raw, dict) and "pairs" in raw:
        doc = GoldenManifestDocument.model_validate(raw)
        return doc.pairs
    raise ValueError(
        "Manifest must be a JSON array of pair entries or an object with a 'pairs' array"
    )


class GoldenDatasetLoader:
    """Read ``data/golden/manifests`` and ``data/golden/annotations``.

    Pass ``golden_root`` pointing at ``data/golden`` (or any parallel layout).
    """

    def __init__(self, golden_root: Path) -> None:
        self._root = golden_root
        self.manifests_dir = golden_root / "manifests"
        self.annotations_dir = golden_root / "annotations"
        self.pairs_dir = golden_root / "pairs"

    def load_manifest_entries(self, name: str) -> list[GoldenPairManifestEntry]:
        path = self.manifests_dir / name
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _parse_manifest_raw(raw)

    def load_manifest_document(self, name: str) -> GoldenManifestDocument:
        """Load a file as a :class:`GoldenManifestDocument` (wraps array-only files)."""
        entries = self.load_manifest_entries(name)
        stem = Path(name).stem
        return GoldenManifestDocument(manifest_id=stem, pairs=entries)

    def load_annotation(self, pair_id: str) -> GoldenPairAnnotation:
        path = self.annotations_dir / f"{pair_id}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        return GoldenPairAnnotation.model_validate(raw)

    def load_annotation_for_entry(
        self,
        entry: GoldenPairManifestEntry,
        *,
        project_root: Path,
    ) -> GoldenPairAnnotation | None:
        if entry.annotation_path:
            p = Path(entry.annotation_path)
            if not p.is_absolute():
                p = project_root / p
            if p.is_file():
                raw = json.loads(p.read_text(encoding="utf-8"))
                return GoldenPairAnnotation.model_validate(raw)
            return None
        path = self.annotations_dir / f"{entry.pair_id}.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return GoldenPairAnnotation.model_validate(raw)
        return None

    def load_index(self) -> GoldenManifestIndex | None:
        path = self.manifests_dir / "index.json"
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return GoldenManifestIndex.model_validate(raw)

    def resolve_pair(
        self,
        entry: GoldenPairManifestEntry,
        project_root: Path,
    ) -> ResolvedGoldenPair:
        a = Path(entry.drawing_a_ref)
        b = Path(entry.drawing_b_ref)
        if not a.is_absolute():
            a = project_root / a
        if not b.is_absolute():
            b = project_root / b
        ann: Path | None = None
        if entry.annotation_path:
            ann = Path(entry.annotation_path)
            if not ann.is_absolute():
                ann = project_root / ann
        elif (self.annotations_dir / f"{entry.pair_id}.json").is_file():
            ann = self.annotations_dir / f"{entry.pair_id}.json"
        return ResolvedGoldenPair(
            pair_id=entry.pair_id,
            drawing_a=a,
            drawing_b=b,
            annotation=ann,
            entry=entry,
        )

    def iter_resolved(
        self,
        manifest_name: str,
        project_root: Path,
    ) -> list[ResolvedGoldenPair]:
        return [self.resolve_pair(e, project_root) for e in self.load_manifest_entries(manifest_name)]

    def load_manifest(self, name: str) -> list[GoldenPairManifestEntry]:
        """Alias for :meth:`load_manifest_entries` (older call sites)."""
        return self.load_manifest_entries(name)
