"""Batch comparison: baseline vs many candidates, single stapled annotated PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger
from app.models.comparison import BatchCompareResponse, BatchPairResult, CompareResponse
from app.preprocessing.settings import PreprocessConfig
from app.reporting.reviewer_bundle import ReviewerVisualKind, build_reviewer_bundle
from app.reporting.visual_diff import merge_reviewer_visual_bytes
from app.services.compare import DrawingCompareService

logger = get_logger(__name__)


def _pair_extras_from_response(response: CompareResponse) -> dict[str, Any]:
    """Extras for API consumer; omit fields that are re-stated on BatchPairResult."""
    skip = frozenset({"comparison_json_report", "comparison_markdown_report"})
    return {k: v for k, v in response.extras.items() if k not in skip}


def run_baseline_many_paths(
    svc: DrawingCompareService,
    baseline: Path,
    candidates: list[Path],
    labels: list[str | None] | None,
    *,
    job_metadata: dict[str, Any] | None,
    fail_fast: bool,
    include_text_reports: bool,
    ocr_provider: str | None,
    preprocess_config: PreprocessConfig | None,
) -> tuple[BatchCompareResponse, bytes, bytes, dict[str, Any]]:
    """Run pairwise compares and build merged PDF + last successful DOCX.

    Returns ``(batch_response, merged_pdf_bytes, comparison_docx_bytes, merged_llm_usage)``.
    """
    meta = dict(job_metadata or {})
    pair_results: list[BatchPairResult] = []
    visual_staple: list[tuple[bytes, ReviewerVisualKind]] = []
    last_docx: bytes | None = None
    merged_llm_usage: dict[str, Any] = {}
    baseline_s = str(baseline.resolve())

    for idx, cand_path in enumerate(candidates):
        cand_uri = str(cand_path.resolve())
        label = None
        if labels is not None:
            label = labels[idx]
        try:
            response, artifacts = svc.compare_paths_with_artifacts(
                baseline,
                cand_path,
                ocr_provider=ocr_provider,
                preprocess_config=preprocess_config,
                job_metadata=meta,
                include_text_reports=include_text_reports,
            )
        except Exception:
            logger.exception("batch pair failed baseline=%s candidate=%s", baseline, cand_path)
            err_msg = "Pairwise compare failed; see server logs"
            pair_results.append(
                BatchPairResult(
                    candidate_uri=cand_uri,
                    candidate_label=label,
                    ok=False,
                    error=err_msg,
                ),
            )
            if fail_fast:
                raise
            continue

        bundle = build_reviewer_bundle(response, artifacts)
        visual_staple.append((bundle.visual_bytes, bundle.visual_kind))
        last_docx = bundle.comparison_docx

        usage = response.extras.get("comparison_llm_usage")
        if isinstance(usage, dict):
            merged_llm_usage.update(usage)

        text_extra: dict[str, Any] = {}
        if include_text_reports:
            for key in ("comparison_json_report", "comparison_markdown_report"):
                if key in response.extras:
                    text_extra[key] = response.extras[key]

        pair_extras = {**_pair_extras_from_response(response), **text_extra}

        pair_results.append(
            BatchPairResult(
                candidate_uri=cand_uri,
                candidate_label=label,
                ok=True,
                comparison_id=response.comparison_id,
                report=response.report,
                review_flags=response.review_flags,
                pair_extras=pair_extras,
            ),
        )

    succeeded = sum(1 for p in pair_results if p.ok)
    if succeeded == 0:
        msg = "All pairwise comparisons in the batch failed"
        raise ValueError(msg)

    merged_pdf = merge_reviewer_visual_bytes(visual_staple)
    if last_docx is None:
        msg = "Internal error: no DOCX produced for batch"
        raise RuntimeError(msg)

    pairs_succeeded = succeeded
    batch_id = uuid4()
    extras: dict[str, Any] = {
        "batch_mode": "baseline_many",
        "pair_count": len(candidates),
        "pairs_succeeded": pairs_succeeded,
        "baseline_path": baseline_s,
    }
    batch_response = BatchCompareResponse(
        batch_id=batch_id,
        baseline_uri=baseline_s,
        pair_count=len(candidates),
        pairs_succeeded=pairs_succeeded,
        pairs=pair_results,
        extras=extras,
    )
    return batch_response, merged_pdf, last_docx, merged_llm_usage
