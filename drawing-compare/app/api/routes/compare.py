"""Drawing comparison: multipart upload (primary) and JSON path-based (server files)."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_compare_service
from app.api.uploads import save_upload
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.comparison import CompareRequest, CompareResponse
from app.ocr.document_provider import OCRError
from app.preprocessing.settings import PreprocessConfig
from app.services.compare import DrawingCompareService

router = APIRouter(tags=["compare"])
logger = get_logger(__name__)


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
    svc: DrawingCompareService = Depends(get_compare_service),  # noqa: B008
) -> CompareResponse:
    preprocess_merged = _parse_preprocess_form(svc, preprocess_config)
    meta = _parse_job_metadata(job_metadata)

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
                svc.compare_paths,
                src_path,
                tgt_path,
                ocr_provider=ocr_provider,
                preprocess_config=preprocess_merged,
                job_metadata=meta,
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
        return svc.compare(body)
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
