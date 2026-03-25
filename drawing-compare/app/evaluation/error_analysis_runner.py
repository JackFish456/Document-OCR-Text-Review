"""Batch error analysis over the golden dataset (uses compare + diagnostics)."""

from __future__ import annotations

import uuid
from collections import Counter
from pathlib import Path

from app.api.deps import build_compare_service
from app.datasets.manifest import GoldenDatasetLoader
from app.evaluation.error_categorize import categorize_field_failure
from app.evaluation.metrics import prediction_for_field
from app.models.diagnostics import PipelineDiagnostics
from app.models.error_analysis import (
    ErrorAnalysisReport,
    ErrorCategory,
    FieldFailureRecord,
    PipelineFailureRecord,
)
from app.models.evaluation import GoldenPairAnnotation
from app.models.match import MatchResult
from app.services.compare import DrawingCompareService


def collect_failures_for_pair(
    pair_id: str,
    ann: GoldenPairAnnotation,
    preds: list[MatchResult],
    diagnostics: PipelineDiagnostics,
    *,
    ocr_confidence_threshold: float,
) -> list[FieldFailureRecord]:
    out: list[FieldFailureRecord] = []
    for exp in ann.fields:
        pred = prediction_for_field(preds, exp.field_id)
        rec = categorize_field_failure(
            pair_id,
            exp.field_id,
            exp.expected,
            pred,
            diagnostics,
            ocr_confidence_threshold=ocr_confidence_threshold,
        )
        if rec is not None:
            out.append(rec)
    return out


def build_error_analysis_report(
    failures: list[FieldFailureRecord],
    *,
    run_id: str,
    manifest_name: str,
    project_root: Path,
    golden_root: Path,
    ocr_provider: str | None,
    ocr_confidence_threshold: float,
    pairs_manifest_no_annotation: int,
    pairs_pipeline_failed: int,
    pipeline_failures: list[PipelineFailureRecord],
    pairs_succeeded: int,
    total_gold_fields_scored: int,
    max_examples_per_category: int = 8,
    top_patterns_limit: int = 20,
) -> ErrorAnalysisReport:
    by_cat: Counter[str] = Counter()
    for f in failures:
        by_cat[f.category.value] += 1

    pattern_counts: dict[str, int] = Counter(f.failure_pattern for f in failures)
    top_patterns = sorted(pattern_counts.items(), key=lambda x: (-x[1], x[0]))[:top_patterns_limit]

    examples: dict[str, list[FieldFailureRecord]] = {c.value: [] for c in ErrorCategory}
    for f in failures:
        bucket = f.category.value
        if len(examples[bucket]) < max_examples_per_category:
            examples[bucket].append(f)

    pairs_with_failures = len({f.pair_id for f in failures})

    return ErrorAnalysisReport(
        run_id=run_id,
        manifest_name=manifest_name,
        project_root=str(project_root.resolve()),
        golden_root=str(golden_root.resolve()),
        ocr_provider=ocr_provider,
        ocr_confidence_threshold=ocr_confidence_threshold,
        pairs_manifest_no_annotation=pairs_manifest_no_annotation,
        pairs_pipeline_failed=pairs_pipeline_failed,
        pipeline_failures=pipeline_failures,
        pairs_succeeded=pairs_succeeded,
        pairs_with_failures=pairs_with_failures,
        total_gold_fields=total_gold_fields_scored,
        total_failures=len(failures),
        failures_by_category=dict(by_cat),
        pattern_counts=dict(pattern_counts),
        top_failure_patterns=top_patterns,
        examples_by_category=examples,
        failures=failures,
    )


def run_batch_error_analysis(
    manifest_name: str,
    *,
    project_root: Path,
    golden_root: Path | None = None,
    run_id: str | None = None,
    ocr_provider: str | None = None,
    ocr_confidence_threshold: float = 0.55,
    service: DrawingCompareService | None = None,
    max_examples_per_category: int = 8,
) -> ErrorAnalysisReport:
    root = project_root.resolve()
    groot = (golden_root or (root / "data" / "golden")).resolve()
    rid = run_id or f"err-{uuid.uuid4().hex[:12]}"
    svc = service or build_compare_service()

    loader = GoldenDatasetLoader(groot)
    entries = loader.load_manifest_entries(manifest_name)
    all_failures: list[FieldFailureRecord] = []
    pipeline_failures: list[PipelineFailureRecord] = []
    succeeded = 0
    total_gold = 0
    no_ann = 0

    for entry in entries:
        ann = loader.load_annotation_for_entry(entry, project_root=root)
        if ann is None or not ann.fields:
            no_ann += 1
            continue

        resolved = loader.resolve_pair(entry, project_root=root)
        try:
            resp, diag = svc.compare_paths_with_diagnostics(
                resolved.drawing_a,
                resolved.drawing_b,
                ocr_provider=ocr_provider,
            )
        except Exception as e:  # noqa: BLE001
            pipeline_failures.append(
                PipelineFailureRecord(pair_id=entry.pair_id, error_message=f"{type(e).__name__}: {e}")
            )
            continue

        succeeded += 1
        total_gold += len(ann.fields)
        preds = resp.report.all_results_flat()
        all_failures.extend(
            collect_failures_for_pair(
                entry.pair_id,
                ann,
                preds,
                diag,
                ocr_confidence_threshold=ocr_confidence_threshold,
            )
        )

    return build_error_analysis_report(
        all_failures,
        run_id=rid,
        manifest_name=manifest_name,
        project_root=root,
        golden_root=groot,
        ocr_provider=ocr_provider,
        ocr_confidence_threshold=ocr_confidence_threshold,
        pairs_manifest_no_annotation=no_ann,
        pairs_pipeline_failed=len(pipeline_failures),
        pipeline_failures=pipeline_failures,
        pairs_succeeded=succeeded,
        total_gold_fields_scored=total_gold,
        max_examples_per_category=max_examples_per_category,
    )
