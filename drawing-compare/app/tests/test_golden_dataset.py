"""Golden dataset schema, loader, validator, and metrics mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.datasets.manifest import GoldenDatasetLoader, ResolvedGoldenPair
from app.datasets.validator import (
    validate_golden_dataset,
    validate_index,
    validate_manifest_entries,
)
from app.evaluation.metrics import (
    prediction_matches_golden,
    score_field_predictions,
)
from app.models.evaluation import (
    GoldenFieldExpectation,
    GoldenManifestIndex,
    GoldenPairAnnotation,
    GoldenPairManifestEntry,
)
from app.models.extraction import ExtractedField
from app.models.golden import GoldenExpectedLabel
from app.models.match import MatchResult, MatchType
from app.models.ocr import BoundingBox


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_golden_expected_label_on_field() -> None:
    row = GoldenFieldExpectation(field_id="x", expected=GoldenExpectedLabel.MATCHED)
    assert row.expected == "matched"


def test_legacy_engine_label_coerced() -> None:
    row = GoldenFieldExpectation.model_validate(
        {"field_id": "x", "expected": "changed_value"},
    )
    assert row.expected == GoldenExpectedLabel.CHANGED


def test_prediction_matches_golden() -> None:
    assert prediction_matches_golden(MatchType.EXACT_MATCH, GoldenExpectedLabel.MATCHED)
    assert prediction_matches_golden(MatchType.PARTIAL_MATCH, GoldenExpectedLabel.MATCHED)
    assert prediction_matches_golden(MatchType.CHANGED_VALUE, GoldenExpectedLabel.CHANGED)
    assert prediction_matches_golden(MatchType.MISSING_IN_TARGET, GoldenExpectedLabel.MISSING)
    assert prediction_matches_golden(MatchType.EXTRA_IN_TARGET, GoldenExpectedLabel.EXTRA)
    assert not prediction_matches_golden(MatchType.UNCERTAIN, GoldenExpectedLabel.MATCHED)


def test_loader_array_and_wrapped_manifest() -> None:
    root = _repo_root()
    golden = root / "data" / "golden"
    loader = GoldenDatasetLoader(golden)
    arr = loader.load_manifest_entries("example_manifest.json")
    assert len(arr) == 1
    assert arr[0].pair_id == "example_pair"
    wrapped = loader.load_manifest_entries("example_wrapped.json")
    assert wrapped[0].pair_id == "example_pair"
    doc = loader.load_manifest_document("example_manifest.json")
    assert doc.manifest_id == "example_manifest"
    assert len(doc.pairs) == 1


def test_loader_annotation_and_resolve() -> None:
    root = _repo_root()
    golden = root / "data" / "golden"
    loader = GoldenDatasetLoader(golden)
    ann = loader.load_annotation("example_pair")
    assert ann.pair_id == "example_pair"
    assert ann.fields[0].expected == GoldenExpectedLabel.MATCHED
    entry = loader.load_manifest_entries("example_manifest.json")[0]
    ann2 = loader.load_annotation_for_entry(entry, project_root=root)
    assert ann2 is not None
    assert ann2.pair_id == "example_pair"
    resolved = loader.resolve_pair(entry, project_root=root)
    assert isinstance(resolved, ResolvedGoldenPair)
    assert resolved.drawing_a.name == "example_a.png"


def test_validate_golden_example_manifest() -> None:
    root = _repo_root()
    golden = root / "data" / "golden"
    loader = GoldenDatasetLoader(golden)
    result = validate_golden_dataset(
        loader,
        project_root=root,
        manifest_name="example_manifest.json",
        check_files_exist=True,
        load_annotations=True,
    )
    assert result.ok, result.errors


def test_validate_index() -> None:
    root = _repo_root()
    idx_path = root / "data" / "golden" / "manifests" / "index.json"
    raw = idx_path.read_text(encoding="utf-8")
    idx = GoldenManifestIndex.model_validate_json(raw)
    res = validate_index(idx, root / "data" / "golden" / "manifests")
    assert res.ok, res.errors


def test_validate_manifest_duplicate_pair_id() -> None:
    entries = [
        GoldenPairManifestEntry(
            pair_id="p1",
            drawing_a_ref="a.png",
            drawing_b_ref="b.png",
        ),
        GoldenPairManifestEntry(
            pair_id="p1",
            drawing_a_ref="a2.png",
            drawing_b_ref="b2.png",
        ),
    ]
    res = validate_manifest_entries(entries, project_root=Path.cwd(), check_files_exist=False)
    assert not res.ok


def test_score_field_predictions() -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=1, y2=1)
    gold = GoldenPairAnnotation(
        pair_id="g1",
        fields=[
            GoldenFieldExpectation(field_id="f1", expected=GoldenExpectedLabel.MATCHED),
            GoldenFieldExpectation(field_id="f2", expected=GoldenExpectedLabel.CHANGED),
        ],
    )
    src1 = ExtractedField(field_id="f1", value="x", bbox=bbox, confidence=1.0)
    tgt1 = ExtractedField(field_id="t1", value="x", bbox=bbox, confidence=1.0)
    src2 = ExtractedField(field_id="f2", value="a", bbox=bbox, confidence=1.0)
    tgt2 = ExtractedField(field_id="t2", value="b", bbox=bbox, confidence=1.0)
    preds = [
        MatchResult(
            source_field=src1,
            target_field=tgt1,
            match_type=MatchType.EXACT_MATCH,
            confidence=1.0,
        ),
        MatchResult(
            source_field=src2,
            target_field=tgt2,
            match_type=MatchType.CHANGED_VALUE,
            confidence=1.0,
        ),
    ]
    metrics = score_field_predictions(gold, preds)
    summary = metrics.summary()
    assert summary.metrics_by_label["matched"].true_positives == 1
    assert summary.metrics_by_label["changed"].true_positives == 1


def test_parse_manifest_rejects_invalid() -> None:
    from app.datasets.manifest import _parse_manifest_raw

    with pytest.raises(ValueError):
        _parse_manifest_raw({"not_pairs": []})
