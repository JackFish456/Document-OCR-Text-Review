"""End-to-end drawing comparison orchestration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.matching.field_comparison_engine import FieldComparisonConfig, FieldComparisonEngine
from app.models.comparison import CompareRequest, CompareResponse
from app.models.diagnostics import PipelineDiagnostics
from app.models.match import MatchResult
from app.models.ocr import OCRPage, OcrRegion
from app.ocr.base import OcrProvider
from app.ocr.document_provider import OCRProvider as DocumentOCRProvider
from app.ocr.factory import get_document_ocr_provider, get_ocr_provider
from app.ocr.geometry_utils import bbox_from_xywh
from app.parsing.fields import RegionParser
from app.preprocessing.io import is_pdf_path
from app.preprocessing.ocr_run import run_ocr_on_preprocessed_page
from app.preprocessing.pipeline import DrawingPreprocessPipeline
from app.preprocessing.settings import PreprocessConfig
from app.preprocessing.types import PreprocessPageResult
from app.reporting.builder import ComparisonReportBuilder
from app.services.compare_artifacts import CompareArtifacts

logger = get_logger(__name__)
_DIRECT_PDF_DOCUMENT_OCR_KEYS = frozenset({"windows", "windows_ocr"})
_MIN_BBOX_EXTENT = 1e-3


def _uses_direct_pdf_document_ocr(settings: Settings, path: Path) -> bool:
    return (
        settings.ocr_provider.lower().strip() in _DIRECT_PDF_DOCUMENT_OCR_KEYS
        and is_pdf_path(path)
    )


def _regions_from_document_page(page: OCRPage) -> list[OcrRegion]:
    out: list[OcrRegion] = []
    if page.lines:
        for line in page.lines:
            text = (line.text or "").strip()
            if not text:
                continue
            out.append(
                OcrRegion(
                    text=text,
                    confidence=float(line.confidence),
                    bbox=line.bbox,
                    page_number=page.page_number,
                )
            )
        return out
    for token in page.tokens:
        text = (token.text or "").strip()
        if not text:
            continue
        out.append(
            OcrRegion(
                text=text,
                confidence=float(token.confidence),
                bbox=token.bbox,
                page_number=page.page_number,
            )
        )
    return out


def _rotate_bound_points(
    points: list[tuple[float, float]],
    *,
    width: float,
    height: float,
    angle_deg: float,
) -> tuple[list[tuple[float, float]], float, float]:
    theta = math.radians(angle_deg)
    alpha = math.cos(theta)
    beta = math.sin(theta)
    cx = width / 2.0
    cy = height / 2.0
    m00 = alpha
    m01 = beta
    m02 = (1.0 - alpha) * cx - beta * cy
    m10 = -beta
    m11 = alpha
    m12 = beta * cx + (1.0 - alpha) * cy
    new_w = float(round(height * abs(m01) + width * abs(m00)))
    new_h = float(round(height * abs(m00) + width * abs(m01)))
    m02 += (new_w / 2.0) - cx
    m12 += (new_h / 2.0) - cy
    return (
        [(m00 * x + m01 * y + m02, m10 * x + m11 * y + m12) for x, y in points],
        new_w,
        new_h,
    )


def _remap_region_bbox_to_preprocessed_page(
    region: OcrRegion,
    *,
    document_page: OCRPage,
    preprocess_page: PreprocessPageResult,
) -> OcrRegion:
    orig_h, orig_w = preprocess_page.original_bgr.shape[:2]
    proc_h, proc_w = preprocess_page.processed_bgr.shape[:2]
    scale_to_orig_x = float(orig_w) / float(document_page.width)
    scale_to_orig_y = float(orig_h) / float(document_page.height)
    bbox = region.bbox
    points = [
        (bbox.x1 * scale_to_orig_x, bbox.y1 * scale_to_orig_y),
        (bbox.x2 * scale_to_orig_x, bbox.y1 * scale_to_orig_y),
        (bbox.x2 * scale_to_orig_x, bbox.y2 * scale_to_orig_y),
        (bbox.x1 * scale_to_orig_x, bbox.y2 * scale_to_orig_y),
    ]
    working_w = float(orig_w)
    working_h = float(orig_h)
    if abs(preprocess_page.deskew_angle_deg) >= 0.05:
        points, working_w, working_h = _rotate_bound_points(
            points,
            width=working_w,
            height=working_h,
            angle_deg=preprocess_page.deskew_angle_deg,
        )
    scale_to_proc_x = float(proc_w) / working_w
    scale_to_proc_y = float(proc_h) / working_h
    scaled = [(x * scale_to_proc_x, y * scale_to_proc_y) for x, y in points]
    xs = [x for x, _ in scaled]
    ys = [y for _, y in scaled]
    x1 = max(0.0, min(min(xs), float(proc_w) - _MIN_BBOX_EXTENT))
    y1 = max(0.0, min(min(ys), float(proc_h) - _MIN_BBOX_EXTENT))
    x2 = max(x1 + _MIN_BBOX_EXTENT, min(max(xs), float(proc_w)))
    y2 = max(y1 + _MIN_BBOX_EXTENT, min(max(ys), float(proc_h)))
    return region.model_copy(
        update={
            "bbox": bbox_from_xywh(x1, y1, x2 - x1, y2 - y1),
            "page_number": preprocess_page.page_number,
        }
    )


def _document_regions_for_preprocessed_page(
    provider: DocumentOCRProvider,
    path: Path,
    preprocess_page: PreprocessPageResult,
) -> list[OcrRegion]:
    doc = provider.extract(str(path))
    page = next((p for p in doc.pages if p.page_number == preprocess_page.page_number), None)
    if page is None:
        return []
    return [
        _remap_region_bbox_to_preprocessed_page(
            region,
            document_page=page,
            preprocess_page=preprocess_page,
        )
        for region in _regions_from_document_page(page)
    ]


class DrawingCompareService:
    """Coordinates preprocessing → OCR → parse → match → classify → report."""

    def __init__(
        self,
        settings: Settings,
        parser: RegionParser,
        reporter: ComparisonReportBuilder,
    ) -> None:
        self._settings = settings
        self._parser = parser
        self._reporter = reporter

    def compare(self, request: CompareRequest) -> CompareResponse:
        path_a = self._resolve_local_path(request.drawing_a_uri)
        path_b = self._resolve_local_path(request.drawing_b_uri)

        if path_a is None or path_b is None:
            logger.info("compare stub: missing local paths for one or both drawings")
            empty: list[MatchResult] = []
            return self._reporter.build(
                empty,
                total_source=0,
                extras={
                    "mode": "stub",
                    "message": (
                        "Use POST /compare with source_file and target_file uploads, "
                        "or POST /compare/paths with drawing_a_uri / drawing_b_uri "
                        "pointing at files on this server."
                    ),
                    "request": request.model_dump(mode="json"),
                },
                include_text_reports=True,
            )

        return self.compare_paths(
            path_a,
            path_b,
            job_metadata=request.job_metadata,
        )

    def merge_preprocess_patch(self, patch: dict[str, Any]) -> PreprocessConfig:
        """Merge a JSON object over the configured defaults (unknown keys rejected)."""
        unknown = set(patch) - set(PreprocessConfig.model_fields)
        if unknown:
            msg = f"Unknown preprocess_config keys: {sorted(unknown)}"
            raise ValueError(msg)
        base = self._settings.preprocess.model_dump()
        base.update(patch)
        return PreprocessConfig.model_validate(base)

    def compare_paths(
        self,
        source: Path,
        target: Path,
        *,
        ocr_provider: str | None = None,
        preprocess_config: PreprocessConfig | None = None,
        job_metadata: dict[str, Any] | None = None,
    ) -> CompareResponse:
        """Run preprocess → OCR → parse → match → classify → reporting for two local paths."""
        response, _ = self.compare_paths_with_artifacts(
            source,
            target,
            ocr_provider=ocr_provider,
            preprocess_config=preprocess_config,
            job_metadata=job_metadata,
        )
        return response

    def compare_paths_with_artifacts(
        self,
        source: Path,
        target: Path,
        *,
        ocr_provider: str | None = None,
        preprocess_config: PreprocessConfig | None = None,
        job_metadata: dict[str, Any] | None = None,
    ) -> tuple[CompareResponse, CompareArtifacts]:
        """Run compare and retain page rasters plus raw match rows for local renderers."""
        eff = self._effective_settings(
            ocr_provider=ocr_provider,
            preprocess_config=preprocess_config,
        )
        logger.info(
            "pipeline start source=%s target=%s ocr_provider=%s",
            source,
            target,
            eff.ocr_provider,
        )
        try:
            artifacts = self._run_pipeline(source, target, settings=eff)
        except Exception:
            logger.exception("pipeline failed source=%s target=%s", source, target)
            raise
        logger.info(
            "pipeline done result_rows=%s total_source=%s",
            len(artifacts.results),
            artifacts.total_source,
        )
        extras: dict[str, Any] = {
            "mode": "full",
            "source_path": str(source.resolve()),
            "target_path": str(target.resolve()),
            "ocr_provider": eff.ocr_provider,
            "job_metadata": job_metadata or {},
        }
        response = self._reporter.build(
            artifacts.results,
            total_source=artifacts.total_source,
            extras=extras,
            include_text_reports=True,
        )
        return response, artifacts

    def compare_paths_with_diagnostics(
        self,
        source: Path,
        target: Path,
        *,
        ocr_provider: str | None = None,
        preprocess_config: PreprocessConfig | None = None,
        job_metadata: dict[str, Any] | None = None,
    ) -> tuple[CompareResponse, PipelineDiagnostics]:
        """Same as :meth:`compare_paths` but also returns parsed fields for tooling / eval."""
        response, artifacts = self.compare_paths_with_artifacts(
            source,
            target,
            ocr_provider=ocr_provider,
            preprocess_config=preprocess_config,
            job_metadata=job_metadata,
        )
        diagnostics = PipelineDiagnostics(
            source_fields=artifacts.source_fields,
            target_fields=artifacts.target_fields,
        )
        return response, diagnostics

    def _effective_settings(
        self,
        *,
        ocr_provider: str | None,
        preprocess_config: PreprocessConfig | None,
    ) -> Settings:
        if not (ocr_provider and ocr_provider.strip()) and preprocess_config is None:
            return self._settings
        updates: dict[str, Any] = {}
        if ocr_provider and ocr_provider.strip():
            updates["ocr_provider"] = ocr_provider.strip().lower()
        if preprocess_config is not None:
            updates["preprocess"] = preprocess_config
        return self._settings.model_copy(update=updates)

    def _resolve_local_path(self, ref: str | None) -> Path | None:
        if not ref:
            return None
        p = Path(ref)
        return p if p.is_file() else None

    def _run_pipeline(
        self,
        path_a: Path,
        path_b: Path,
        *,
        settings: Settings,
    ) -> CompareArtifacts:
        preprocessor = DrawingPreprocessPipeline(settings)
        pages_a = preprocessor.process_path(path_a)
        pages_b = preprocessor.process_path(path_b)
        if len(pages_a) > 1:
            logger.warning("Drawing A has %s pages; using page 1 only for compare", len(pages_a))
        if len(pages_b) > 1:
            logger.warning("Drawing B has %s pages; using page 1 only for compare", len(pages_b))
        pa = pages_a[0]
        pb = pages_b[0]
        ndarray_ocr: OcrProvider | None = None
        document_ocr: DocumentOCRProvider | None = None

        def _get_ndarray_ocr() -> OcrProvider:
            nonlocal ndarray_ocr
            if ndarray_ocr is None:
                ndarray_ocr = get_ocr_provider(settings)
            return ndarray_ocr

        def _get_document_ocr() -> DocumentOCRProvider:
            nonlocal document_ocr
            if document_ocr is None:
                bridge_settings = settings.model_copy(
                    update={"document_ocr_provider": settings.ocr_provider}
                )
                document_ocr = get_document_ocr_provider(bridge_settings)
            return document_ocr

        def _ocr_regions(path: Path, page: PreprocessPageResult) -> list[OcrRegion]:
            if _uses_direct_pdf_document_ocr(settings, path):
                provider = _get_document_ocr()
                logger.info(
                    "using direct PDF document OCR source=%s provider=%s page=%s",
                    path,
                    provider.provider_name,
                    page.page_number,
                )
                return _document_regions_for_preprocessed_page(provider, path, page)
            return run_ocr_on_preprocessed_page(_get_ndarray_ocr(), page)

        regions_a = _ocr_regions(path_a, pa)
        regions_b = _ocr_regions(path_b, pb)
        fields_a = self._parser.parse(regions_a).fields
        fields_b = self._parser.parse(regions_b).fields
        engine_cfg = FieldComparisonConfig.from_settings_like(
            fuzzy_match_threshold=settings.fuzzy_match_threshold,
            fuzzy_changed_threshold=settings.fuzzy_changed_threshold,
            uncertain_score_low=settings.uncertain_score_low,
        )
        engine = FieldComparisonEngine(engine_cfg)
        results = engine.build_match_results(fields_a, fields_b)
        return CompareArtifacts(
            source_page=pa,
            target_page=pb,
            results=results,
            source_fields=fields_a,
            target_fields=fields_b,
        )
