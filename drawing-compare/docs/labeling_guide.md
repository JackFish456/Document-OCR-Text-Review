# Golden dataset labeling guide

This project evaluates OCR-based drawing comparison using **labeled drawing pairs**. Ground truth lives under `data/golden/`:

| Path | Purpose |
|------|---------|
| `pairs/` | Source PDFs or images for drawing A and drawing B |
| `annotations/` | One JSON file per pair: field-level expected labels |
| `manifests/` | Lists of pairs, paths, and optional registry `index.json` |

To validate on disk (from the `drawing-compare` repo root, with the package installed or `PYTHONPATH=.`):

```bash
python -m app.datasets --manifest example_manifest.json
```

Use `--no-check-files` if drawings are not yet copied locally.

## Drawing pair (manifest row)

Each pair compares **drawing A** (baseline / older) to **drawing B** (target / newer). Add a row to a manifest file (see formats below) with:

- `pair_id` — stable string; must match `pair_id` inside the annotation JSON
- `drawing_a_ref`, `drawing_b_ref` — paths relative to the **repository root** (recommended) or absolute paths
- `annotation_path` — optional; defaults to `data/golden/annotations/{pair_id}.json` when omitted
- `notes` — optional; document why the two sheets belong together

## Field-level labels (`expected`)

Each labeled field uses one of four **golden** values (stable for metrics and dataset growth):

| Value | Meaning |
|-------|---------|
| `matched` | Same semantic content on A and B (minor OCR or spacing differences are OK if meaning is the same). |
| `changed` | Same role or location, but the value meaningfully differs between drawings. |
| `missing` | Present on drawing A, absent on drawing B (after accounting for crop or sheet scope). |
| `extra` | Present on drawing B, not on drawing A. |

**Alignment with the engine:** at runtime the comparer emits finer `MatchType` values (`exact_match`, `partial_match`, `changed_value`, `missing_in_target`, `extra_in_target`, `uncertain`). Evaluation maps those onto the four golden labels (for example `exact_match` and `partial_match` both satisfy `matched`). Legacy annotation files may still use engine strings such as `changed_value`; they are normalized on load to the golden vocabulary.

## Annotation file shape

Each `data/golden/annotations/{pair_id}.json` file follows `GoldenPairAnnotation`:

- `schema_version` — integer; bump when you make breaking annotation changes
- `pair_id` — must equal the manifest row
- `fields` — list of `{ "field_id", "expected", "optional_notes?" }`
- `pair_level_expected` — optional coarse label for the whole pair (`matched` | `changed` | `missing` | `extra`)
- `review_note` — optional human note for ambiguous cases
- `metadata` — free-form (e.g. `labeled_by`, batch id)

Use **stable `field_id` strings** across exports so time-series metrics stay comparable.

## Manifest formats

**Array (minimal):** a JSON array of `GoldenPairManifestEntry` objects — see `data/golden/manifests/example_manifest.json`.

**Wrapped (recommended for new splits):** a `GoldenManifestDocument` object with `schema_version`, `manifest_id`, optional `description`, and a `pairs` array — see `data/golden/manifests/example_wrapped.json`.

## Expanding the dataset

1. Copy drawings into `data/golden/pairs/` (or keep them elsewhere and reference with consistent relative paths).
2. Add a manifest row (or add to an existing manifest file).
3. Create `annotations/{pair_id}.json` with field-level `expected` labels.
4. Register new manifest files in `data/golden/manifests/index.json` when you introduce train/validation splits.
5. Run `python -m app.datasets --manifest <file>` before committing.

## API reference (code)

- Schemas: `app.models.golden.GoldenExpectedLabel`, `app.models.evaluation.GoldenPairAnnotation`, `GoldenPairManifestEntry`, `GoldenManifestDocument`, `GoldenManifestIndex`
- Loader: `app.datasets.manifest.GoldenDatasetLoader` — constructed with `golden_root=Path("data/golden")`
- Validator: `app.datasets.validator.validate_golden_dataset` and related helpers
