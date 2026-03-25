"""Drawing comparison: multipart upload (primary) and JSON path-based (server files)."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.api.compare_job_store import (
    artifact_file_path,
    media_type_for_filename,
    persist_reviewer_artifacts,
)
from app.api.deps import get_compare_service
from app.api.uploads import save_upload
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.comparison import CompareRequest, CompareResponse
from app.ocr.document_provider import OCRError
from app.preprocessing.settings import PreprocessConfig
from app.reporting.reviewer_bundle import build_reviewer_bundle, reviewer_output_extras
from app.services.compare import DrawingCompareService
from app.services.compare_artifacts import CompareArtifacts

router = APIRouter(tags=["compare"])
logger = get_logger(__name__)


def _parse_include_text_reports_form(raw: str | None) -> bool:
    """Multipart form often sends strings; default True when omitted."""
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in ("false", "0", "no", "off")


def _attach_reviewer_outputs(
    response: CompareResponse,
    compare_artifacts: CompareArtifacts,
) -> CompareResponse:
    bundle = build_reviewer_bundle(response, compare_artifacts)
    job_id = uuid4().hex
    persist_reviewer_artifacts(job_id, bundle)
    merged = dict(response.extras)
    merged.update(reviewer_output_extras(job_id, bundle))
    return response.model_copy(update={"extras": merged})


def _run_compare_with_reviewer_outputs(
    svc: DrawingCompareService,
    src_path: Path,
    tgt_path: Path,
    *,
    ocr_provider: str | None,
    preprocess_merged: PreprocessConfig | None,
    meta: dict[str, Any],
    include_text_reports: bool,
) -> CompareResponse:
    response, artifacts = svc.compare_paths_with_artifacts(
        src_path,
        tgt_path,
        ocr_provider=ocr_provider,
        preprocess_config=preprocess_merged,
        job_metadata=meta,
        include_text_reports=include_text_reports,
    )
    return _attach_reviewer_outputs(response, artifacts)


def _resolve_local_path(ref: str | None) -> Path | None:
    if not ref:
        return None
    path = Path(ref)
    return path if path.is_file() else None


def _upload_temp_root() -> Path:
    root = get_settings().data_dir / "tmp" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def _upload_workspace() -> Path:
    root = _upload_temp_root() / f"drawing_compare_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _parse_job_metadata(raw: str | None) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"job_metadata must be valid JSON: {e}",
        ) from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="job_metadata must be a JSON object")
    return data


def _parse_preprocess_form(
    svc: DrawingCompareService,
    raw: str | None,
) -> PreprocessConfig | None:
    if not raw or not str(raw).strip():
        return None
    try:
        patch = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"preprocess_config must be valid JSON: {e}",
        ) from e
    if not isinstance(patch, dict):
        raise HTTPException(status_code=422, detail="preprocess_config must be a JSON object")
    try:
        return svc.merge_preprocess_patch(patch)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two uploaded drawings",
    description=(
        "Runs preprocess → OCR → normalization → field extraction → matching → "
        "classification → reporting. Returns ComparisonReport and review flags."
    ),
)
async def compare_uploaded_drawings(
    source_file: Annotated[UploadFile, File(description="Source drawing (PDF or raster image)")],
    target_file: Annotated[UploadFile, File(description="Target drawing (PDF or raster image)")],
    ocr_provider: Annotated[
        str | None,
        Form(
            description=(
                "OCR provider: stub | echo | stub_document | tesseract | paddleocr (alias: paddle) "
                "| windows_ocr (alias: windows) (omit to use server default)"
            ),
        ),
    ] = None,
    preprocess_config: Annotated[
        str | None,
        Form(
            description=(
                "Optional JSON object of PreprocessConfig fields "
                "to merge over server defaults"
            ),
        ),
    ] = None,
    job_metadata: Annotated[
        str | None,
        Form(description="Optional JSON object attached to response extras"),
    ] = None,
    include_text_reports: Annotated[
        str | None,
        Form(
            description=(
                'Set to "false" to omit comparison_json_report and comparison_markdown_report '
                "from extras (smaller JSON). Default true."
            ),
        ),
    ] = None,
    svc: DrawingCompareService = Depends(get_compare_service),  # noqa: B008
) -> CompareResponse:
    preprocess_merged = _parse_preprocess_form(svc, preprocess_config)
    meta = _parse_job_metadata(job_metadata)
    include_txt = _parse_include_text_reports_form(include_text_reports)

    with _upload_workspace() as root:
        try:
            src_path = await save_upload(
                source_file,
                dest_dir=root,
                fallback_stem="source",
            )
            tgt_path = await save_upload(
                target_file,
                dest_dir=root,
                fallback_stem="target",
            )
        except HTTPException:
            raise
        except OSError as e:
            logger.exception("failed to persist uploads")
            raise HTTPException(
                status_code=500,
                detail="Could not store uploaded files temporarily",
            ) from e

        try:
            return await run_in_threadpool(
                _run_compare_with_reviewer_outputs,
                svc,
                src_path,
                tgt_path,
                ocr_provider=ocr_provider,
                preprocess_merged=preprocess_merged,
                meta=meta,
                include_text_reports=include_txt,
            )
        except ValueError as e:
            logger.warning("compare request rejected: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except OCRError as e:
            logger.warning("ocr request rejected: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception:
            logger.exception(
                "compare pipeline error source=%s target=%s",
                src_path.name,
                tgt_path.name,
            )
            raise HTTPException(
                status_code=500,
                detail="The comparison pipeline failed; see server logs for details",
            ) from None


@router.get(
    "/compare/artifacts/{job_id}/{filename}",
    summary="Download a persisted compare artifact (PDF, Word, PNG, HTML, or manifest JSON)",
)
def download_compare_artifact(job_id: str, filename: str) -> FileResponse:
    path = artifact_file_path(job_id, filename)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="Artifact not found or expired",
        )
    return FileResponse(
        path,
        filename=filename,
        media_type=media_type_for_filename(filename),
    )


@router.post(
    "/compare/paths",
    response_model=CompareResponse,
    summary="Compare drawings via server-local paths",
)
def compare_from_paths(
    body: CompareRequest,
    svc: DrawingCompareService = Depends(get_compare_service),  # noqa: B008
) -> CompareResponse:
    """Use when both PDFs/images are already on disk (paths visible to the API process)."""
    try:
        source = _resolve_local_path(body.drawing_a_uri)
        target = _resolve_local_path(body.drawing_b_uri)
        if source is None or target is None:
            return svc.compare(body)
        response, artifacts = svc.compare_paths_with_artifacts(
            source,
            target,
            job_metadata=body.job_metadata,
            include_text_reports=True,
        )
        return _attach_reviewer_outputs(response, artifacts)
    except ValueError as e:
        logger.warning("compare/paths request rejected: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OCRError as e:
        logger.warning("compare/paths OCR rejected: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        logger.exception("compare/paths pipeline failure")
        raise HTTPException(
            status_code=500,
            detail="The comparison pipeline failed; see server logs for details",
        ) from None
