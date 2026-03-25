"""Run the full compare pipeline over a golden manifest and aggregate metrics."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.api.deps import build_compare_service
from app.datasets.manifest import GoldenDatasetLoader
from app.evaluation.metrics import EvaluationMetrics, score_field_predictions
from app.evaluation.reporting import compose_batch_report
from app.models.evaluation import BatchEvaluationReport, PairEvaluationRecord
from app.services.compare import DrawingCompareService


def run_batch_golden_evaluation(
    manifest_name: str,
    *,
    project_root: Path,
    golden_root: Path | None = None,
    run_id: str | None = None,
    ocr_provider: str | None = None,
    service: DrawingCompareService | None = None,
) -> BatchEvaluationReport:
    """Execute :meth:`DrawingCompareService.compare_paths` for each manifest row with annotations.

    Predictions are aligned to gold via ``field_id`` on :class:`~app.models.match.MatchResult`
    rows (see :func:`~app.evaluation.metrics.prediction_for_field`).
    """
    root = project_root.resolve()
    groot = (golden_root or (root / "data" / "golden")).resolve()
    rid = run_id or f"eval-{uuid.uuid4().hex[:12]}"
    svc = service or build_compare_service()

    loader = GoldenDatasetLoader(groot)
    entries = loader.load_manifest_entries(manifest_name)
    merged = EvaluationMetrics(run_id=rid)
    pair_results: list[PairEvaluationRecord] = []
    succeeded = 0
    failed = 0

    for entry in entries:
        ann = loader.load_annotation_for_entry(entry, project_root=root)
        if ann is None:
            failed += 1
            pair_results.append(
                PairEvaluationRecord(
                    pair_id=entry.pair_id,
                    success=False,
                    error_message="Annotation file not found or not readable",
                )
            )
            continue
        if not ann.fields:
            failed += 1
            pair_results.append(
                PairEvaluationRecord(
                    pair_id=entry.pair_id,
                    success=False,
                    error_message="Annotation has no fields to score",
                )
            )
            continue

        resolved = loader.resolve_pair(entry, project_root=root)
        try:
            resp = svc.compare_paths(
                resolved.drawing_a,
                resolved.drawing_b,
                ocr_provider=ocr_provider,
            )
        except Exception as e:  # noqa: BLE001 — surface pipeline failures per pair
            failed += 1
            pair_results.append(
                PairEvaluationRecord(
                    pair_id=entry.pair_id,
                    success=False,
                    error_message=f"{type(e).__name__}: {e}",
                )
            )
            continue

        preds = resp.report.all_results_flat()
        m = score_field_predictions(ann, preds)
        merged.merge_from(m)
        succeeded += 1
        pair_results.append(
            PairEvaluationRecord(
                pair_id=entry.pair_id,
                success=True,
                gold_field_count=len(ann.fields),
                fields_correct=m.total_correct(),
                fields_total=m.total_gold_fields(),
            )
        )

    return compose_batch_report(
        merged,
        run_id=rid,
        manifest_name=manifest_name,
        project_root=root,
        golden_root=groot,
        ocr_provider=ocr_provider,
        pair_results=pair_results,
        pairs_attempted=len(entries),
        pairs_succeeded=succeeded,
        pairs_failed=failed,
    )
