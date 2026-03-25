# Evaluation guide

## Goals

- Measure **field-level** agreement between model outputs and golden annotations.
- Track regressions when swapping OCR providers, preprocessing, or thresholds.

## Layout

- `data/golden/manifests/*.json` — lists pairs and paths to drawings + optional annotation paths.
- `data/golden/annotations/*.json` — `GoldenPairAnnotation` payload per pair.
- `experiments/eval_runs/` — persisted metric JSON from batch jobs (populate via your runner script).

## Running metrics (code)

Use `app.evaluation.metrics.EvaluationMetrics` and `score_field_predictions` after you obtain `StructuredComparisonResult` for each pair—for example by calling `DrawingCompareService.compare` in a loop. The reference implementation matches predictions to gold by exact `field_id`; extend alignment if your parser emits synthetic IDs.

### Recommended workflow

1. Freeze a **manifest version** (git tag or manifest checksum).
2. Sweep OCR providers / thresholds into `experiments/provider_benchmarks/` with metadata (Git SHA, dependency versions).
3. Compare **macro-F1** per label plus error taxonomy buckets (see `docs/error_taxonomy.md`).
4. Investigate failures starting from `review_flags` in model output and low `fuzzy_ratio` pairs.

## Configuration

Thresholds in `DRAWING_COMPARE_*` (see `app/core/config.py`) directly affect `changed` vs `matched` vs `uncertain`. Re-tune on a held-out golden split, not on the full set you report.
