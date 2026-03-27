"""End-to-end drawing comparison orchestration."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal

from app.core.config import Settings
from app.core.logging import get_logger
from app.matching.field_comparison_engine import FieldComparisonConfig, FieldComparisonEngine
from app.models.comparison import CompareRequest, CompareResponse
from app.models.diagnostics import PipelineDiagnostics
from app.models.extraction import ExtractedField
from app.models.match import MatchResult
from app.models.ocr import OCRDocument, OCRPage, OcrRegion
from app.ocr.base import OCRProvider, OcrProvider, OCRResult
from app.ocr.factory import get_ocr_provider, get_ocr_result_provider
from app.ocr.geometry_utils import bbox_from_xywh
from app.ocr.result_adapter import ocr_provider_document_label, ocr_result_to_document
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


def _merge_ocr_results(left: OCRResult, right: OCRResult) -> OCRResult:
    """Concatenate pages from two file-level OCR results (pages renumbered sequentially)."""
    n = len(left.pages)
    merged_pages = list(left.pages)
    for i, p in enumerate(right.pages):
        merged_pages.append(p.model_copy(update={"page_number": n + i + 1}))
    return OCRResult(pages=merged_pages)


def _merge_ocr_paths(
    by_path: dict[str, OCRResult],
    path_a: Path,
    path_b: Path,
) -> OCRResult | None:
    """Merge OCR from source/target paths when both used document OCR (PDF cache keys)."""
    ka = str(path_a.resolve())
    kb = str(path_b.resolve())
    ra = by_path.get(ka)
    rb = by_path.get(kb)
    if ra is None and rb is None:
        return None
    if ra is None:
        return rb
    if rb is None:
        return ra
    return _merge_ocr_results(ra, rb)


def _dual_branch_pipeline_payload(
    response: CompareResponse,
    artifacts: CompareArtifacts,
) -> dict[str, Any]:
    """Serializable output for one dual-OCR branch (no raw page rasters)."""
    return {
        "ok": True,
        "compare_response": response.model_dump(mode="json"),
        "artifacts": {
            "results": [r.model_dump(mode="json") for r in artifacts.results],
            "source_fields": [f.model_dump(mode="json") for f in artifacts.source_fields],
            "target_fields": [f.model_dump(mode="json") for f in artifacts.target_fields],
            "source_page_count": len(artifacts.source_pages),
            "target_page_count": len(artifacts.target_pages),
        },
    }


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, object]) -> None:
    # region agent log
    payload = {
        "sessionId": "2f2721",
        "runId": "initial",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        _log_path = Path(
            "C:/Users/Jack.Fisher/OneDrive - Kiewit Corporation/Desktop/"
            "Document OCR Text Review/debug-2f2721.log"
        )
        with _log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


def _uses_direct_pdf_document_ocr(settings: Settings, path: Path) -> bool:
    return (
        settings.ocr_provider.lower().strip() in _DIRECT_PDF_DOCUMENT_OCR_KEYS
        and is_pdf_path(path)
    )


def _stamp_region_page(regions: list[OcrRegion], page_number: int) -> list[OcrRegion]:
    """Ensure every region carries the preprocess page index (some OCR stubs omit it)."""
    return [r.model_copy(update={"page_number": page_number}) for r in regions]


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
    path: Path,
    preprocess_page: PreprocessPageResult,
    *,
    document: OCRDocument,
) -> list[OcrRegion]:
    doc = document
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
        ocr_result_backend: Literal["local", "google"] | None = None,
        enable_dual_ocr: bool | None = None,
    ) -> CompareResponse:
        """Run preprocess → OCR → parse → match → classify → reporting for two local paths."""
        response, _ = self.compare_paths_with_artifacts(
            source,
            target,
            ocr_provider=ocr_provider,
            preprocess_config=preprocess_config,
            job_metadata=job_metadata,
            include_text_reports=True,
            ocr_result_backend=ocr_result_backend,
            enable_dual_ocr=enable_dual_ocr,
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
        include_text_reports: bool = True,
        ocr_result_backend: Literal["local", "google"] | None = None,
        enable_dual_ocr: bool | None = None,
    ) -> tuple[CompareResponse, CompareArtifacts]:
        """Run compare and retain page rasters plus raw match rows for local renderers."""
        eff = self._effective_settings(
            ocr_provider=ocr_provider,
            preprocess_config=preprocess_config,
            ocr_result_backend=ocr_result_backend,
            enable_dual_ocr=enable_dual_ocr,
        )
        logger.info(
            "pipeline start source=%s target=%s ocr_provider=%s",
            source,
            target,
            eff.ocr_provider,
        )
        # region agent log
        _debug_log(
            "H5",
            "app/services/compare.py:compare_paths_with_artifacts:start",
            "Starting compare pipeline",
            {
                "source": str(source),
                "target": str(target),
                "ocr_provider": eff.ocr_provider,
            },
        )
        # endregion
        try:
            if eff.enable_dual_ocr:
                return self._compare_paths_dual_ocr(
                    source,
                    target,
                    eff=eff,
                    job_metadata=job_metadata,
                    include_text_reports=include_text_reports,
                )
            artifacts, _ = self._run_pipeline(source, target, settings=eff)
        except Exception:
            logger.exception("pipeline failed source=%s target=%s", source, target)
            raise
        logger.info(
            "pipeline done result_rows=%s total_source=%s",
            len(artifacts.results),
            artifacts.total_source,
        )
        # region agent log
        _debug_log(
            "H8",
            "app/services/compare.py:compare_paths_with_artifacts:end",
            "Pipeline completed",
            {
                "result_rows": len(artifacts.results),
                "source_fields": len(artifacts.source_fields),
                "target_fields": len(artifacts.target_fields),
            },
        )
        # endregion
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
            include_text_reports=include_text_reports,
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
        ocr_result_backend: Literal["local", "google"] | None = None,
        enable_dual_ocr: bool | None = None,
    ) -> tuple[CompareResponse, PipelineDiagnostics]:
        """Same as :meth:`compare_paths` but also returns parsed fields for tooling / eval."""
        response, artifacts = self.compare_paths_with_artifacts(
            source,
            target,
            ocr_provider=ocr_provider,
            preprocess_config=preprocess_config,
            job_metadata=job_metadata,
            include_text_reports=True,
            ocr_result_backend=ocr_result_backend,
            enable_dual_ocr=enable_dual_ocr,
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
        ocr_result_backend: Literal["local", "google"] | None = None,
        enable_dual_ocr: bool | None = None,
    ) -> Settings:
        # region agent log
        _debug_log(
            "H9",
            "app/services/compare.py:_effective_settings:input",
            "Resolving effective settings",
            {
                "requested_ocr_provider": ocr_provider,
                "base_settings_ocr_provider": self._settings.ocr_provider,
                "has_preprocess_override": preprocess_config is not None,
                "ocr_result_backend": ocr_result_backend,
                "enable_dual_ocr": enable_dual_ocr,
            },
        )
        # endregion
        if (
            not (ocr_provider and ocr_provider.strip())
            and preprocess_config is None
            and ocr_result_backend is None
            and enable_dual_ocr is None
        ):
            # region agent log
            _debug_log(
                "H9",
                "app/services/compare.py:_effective_settings:result",
                "Using base settings",
                {"effective_ocr_provider": self._settings.ocr_provider},
            )
            # endregion
            return self._settings
        updates: dict[str, Any] = {}
        if ocr_provider and ocr_provider.strip():
            updates["ocr_provider"] = ocr_provider.strip().lower()
        if preprocess_config is not None:
            updates["preprocess"] = preprocess_config
        if ocr_result_backend is not None:
            updates["ocr_result_backend"] = ocr_result_backend
        if enable_dual_ocr is not None:
            updates["enable_dual_ocr"] = enable_dual_ocr
        resolved = self._settings.model_copy(update=updates)
        # region agent log
        _debug_log(
            "H9",
            "app/services/compare.py:_effective_settings:result",
            "Using merged settings",
            {"effective_ocr_provider": resolved.ocr_provider},
        )
        # endregion
        return resolved

    def _resolve_local_path(self, ref: str | None) -> Path | None:
        if not ref:
            return None
        p = Path(ref)
        return p if p.is_file() else None

    def _compare_paths_dual_ocr(
        self,
        source: Path,
        target: Path,
        *,
        eff: Settings,
        job_metadata: dict[str, Any] | None,
        include_text_reports: bool,
    ) -> tuple[CompareResponse, CompareArtifacts]:
        """Run local and Google OCR backends independently; failures are isolated per branch."""
        local_response: CompareResponse | None = None
        local_artifacts: CompareArtifacts | None = None
        local_err: str | None = None
        google_response: CompareResponse | None = None
        google_artifacts: CompareArtifacts | None = None
        google_err: str | None = None

        local_ocr_by_path: dict[str, OCRResult] = {}
        google_ocr_by_path: dict[str, OCRResult] = {}
        try:
            local_artifacts, local_ocr_by_path = self._run_pipeline(
                source,
                target,
                settings=eff,
                ocr_result_backend="local",
            )
            local_response = self._reporter.build(
                local_artifacts.results,
                total_source=local_artifacts.total_source,
                extras={
                    "mode": "full",
                    "source_path": str(source.resolve()),
                    "target_path": str(target.resolve()),
                    "ocr_provider": eff.ocr_provider,
                    "ocr_result_backend": "local",
                    "job_metadata": job_metadata or {},
                },
                include_text_reports=include_text_reports,
            )
        except Exception as exc:
            logger.exception("dual OCR local pipeline failed")
            local_err = str(exc)

        try:
            google_artifacts, google_ocr_by_path = self._run_pipeline(
                source,
                target,
                settings=eff,
                ocr_result_backend="google",
            )
            google_response = self._reporter.build(
                google_artifacts.results,
                total_source=google_artifacts.total_source,
                extras={
                    "mode": "full",
                    "source_path": str(source.resolve()),
                    "target_path": str(target.resolve()),
                    "ocr_provider": eff.ocr_provider,
                    "ocr_result_backend": "google",
                    "job_metadata": job_metadata or {},
                },
                include_text_reports=include_text_reports,
            )
        except Exception as exc:
            logger.exception("dual OCR google pipeline failed")
            google_err = str(exc)

        if local_response is None and google_response is None:
            msg = f"dual OCR: both pipelines failed (local={local_err!r}, google={google_err!r})"
            raise RuntimeError(msg) from None

        local_out: dict[str, Any]
        if local_response is not None and local_artifacts is not None:
            local_out = _dual_branch_pipeline_payload(local_response, local_artifacts)
        else:
            local_out = {"ok": False, "error": local_err or "local_pipeline_failed"}

        google_out: dict[str, Any]
        if google_response is not None and google_artifacts is not None:
            google_out = _dual_branch_pipeline_payload(google_response, google_artifacts)
        else:
            google_out = {"ok": False, "error": google_err or "google_pipeline_failed"}

        primary = local_response if local_response is not None else google_response
        assert primary is not None
        artifacts_out = (
            local_artifacts if local_artifacts is not None else google_artifacts
        )
        assert artifacts_out is not None

        ocr_comparison: dict[str, Any]
        try:
            from app.ocr.ocr_comparator import compare_ocr

            merged_local = _merge_ocr_paths(local_ocr_by_path, source, target)
            merged_google = _merge_ocr_paths(google_ocr_by_path, source, target)
            if merged_local is not None and merged_google is not None:
                metrics = compare_ocr(merged_local, merged_google)
                ocr_comparison = {
                    **metrics.model_dump(mode="json"),
                    "summary": metrics.summary(),
                }
            else:
                ocr_comparison = {
                    "ok": False,
                    "reason": "no_pdf_document_ocr_results",
                }
        except Exception as exc:
            logger.exception("dual OCR comparison metrics failed")
            ocr_comparison = {"ok": False, "error": str(exc)}

        combined_extras: dict[str, Any] = {
            "mode": "dual_ocr",
            "source_path": str(source.resolve()),
            "target_path": str(target.resolve()),
            "ocr_provider": eff.ocr_provider,
            "job_metadata": job_metadata or {},
            "local": local_out,
            "google": google_out,
            "ocr_comparison": ocr_comparison,
        }
        response = primary.model_copy(update={"extras": combined_extras})

        logger.info(
            "dual OCR pipeline done local_ok=%s google_ok=%s",
            local_out.get("ok"),
            google_out.get("ok"),
        )
        return response, artifacts_out

    def _run_pipeline(
        self,
        path_a: Path,
        path_b: Path,
        *,
        settings: Settings,
        ocr_result_backend: Literal["local", "google"] | None = None,
    ) -> tuple[CompareArtifacts, dict[str, OCRResult]]:
        pipeline_settings = (
            settings.model_copy(update={"ocr_result_backend": ocr_result_backend})
            if ocr_result_backend is not None
            else settings
        )
        preprocessor = DrawingPreprocessPipeline(settings)
        pages_a = preprocessor.process_path(path_a)
        pages_b = preprocessor.process_path(path_b)
        # region agent log
        _debug_log(
            "H6",
            "app/services/compare.py:_run_pipeline:preprocess",
            "Preprocessing completed",
            {
                "source_pages": len(pages_a),
                "target_pages": len(pages_b),
            },
        )
        # endregion
        n_pairs = max(len(pages_a), len(pages_b))
        if len(pages_a) != len(pages_b):
            logger.info(
                "page count mismatch: source=%s target=%s; aligning by index (N↔N), "
                "extra pages reported as missing/extra fields",
                len(pages_a),
                len(pages_b),
            )

        ndarray_ocr: OcrProvider | None = None
        ocr_result_provider: OCRProvider | None = None
        pdf_doc_cache: dict[str, OCRDocument] = {}
        pdf_ocr_result_cache: dict[str, OCRResult] = {}

        def _get_ndarray_ocr() -> OcrProvider:
            nonlocal ndarray_ocr
            if ndarray_ocr is None:
                ndarray_ocr = get_ocr_provider(settings)
            return ndarray_ocr

        def _get_ocr_result_provider() -> OCRProvider:
            nonlocal ocr_result_provider
            if ocr_result_provider is None:
                bridge_settings = pipeline_settings.model_copy(
                    update={"document_ocr_provider": pipeline_settings.ocr_provider}
                )
                ocr_result_provider = get_ocr_result_provider(bridge_settings)
            return ocr_result_provider

        def _cached_pdf_document(path: Path) -> OCRDocument:
            key = str(path.resolve())
            if key not in pdf_doc_cache:
                prov = _get_ocr_result_provider()
                ocr_result = prov.extract(str(path))
                pdf_ocr_result_cache[key] = ocr_result
                pdf_doc_cache[key] = ocr_result_to_document(
                    ocr_result,
                    document_id=path.stem or path.name or "document",
                    source_path=str(path),
                    provider_name=ocr_provider_document_label(prov),
                )
            return pdf_doc_cache[key]

        def _ocr_regions(path: Path, page: PreprocessPageResult) -> list[OcrRegion]:
            if _uses_direct_pdf_document_ocr(settings, path):
                doc = _cached_pdf_document(path)
                logger.info(
                    "using direct PDF document OCR source=%s provider=%s page=%s",
                    path,
                    doc.provider_name,
                    page.page_number,
                )
                return _document_regions_for_preprocessed_page(path, page, document=doc)
            return run_ocr_on_preprocessed_page(_get_ndarray_ocr(), page)

        fields_a: list[ExtractedField] = []
        fields_b: list[ExtractedField] = []
        for i in range(n_pairs):
            if i < len(pages_a):
                regions_a = _stamp_region_page(
                    _ocr_regions(path_a, pages_a[i]),
                    pages_a[i].page_number,
                )
                parsed_a = self._parser.parse(regions_a).fields
                fields_a.extend(parsed_a)
                # region agent log
                _debug_log(
                    "H6",
                    "app/services/compare.py:_run_pipeline:source_page",
                    "Source OCR and parse counts",
                    {
                        "pair_index": i,
                        "page_number": pages_a[i].page_number,
                        "ocr_regions": len(regions_a),
                        "parsed_fields": len(parsed_a),
                    },
                )
                # endregion
            if i < len(pages_b):
                regions_b = _stamp_region_page(
                    _ocr_regions(path_b, pages_b[i]),
                    pages_b[i].page_number,
                )
                parsed_b = self._parser.parse(regions_b).fields
                fields_b.extend(parsed_b)
                # region agent log
                _debug_log(
                    "H6",
                    "app/services/compare.py:_run_pipeline:target_page",
                    "Target OCR and parse counts",
                    {
                        "pair_index": i,
                        "page_number": pages_b[i].page_number,
                        "ocr_regions": len(regions_b),
                        "parsed_fields": len(parsed_b),
                    },
                )
                # endregion

        engine_cfg = FieldComparisonConfig.from_settings_like(
            fuzzy_match_threshold=settings.fuzzy_match_threshold,
            fuzzy_changed_threshold=settings.fuzzy_changed_threshold,
            uncertain_score_low=settings.uncertain_score_low,
        )
        engine = FieldComparisonEngine(engine_cfg)
        results = engine.build_match_results(fields_a, fields_b)
        # region agent log
        _debug_log(
            "H7",
            "app/services/compare.py:_run_pipeline:matching",
            "Matching output counts",
            {
                "source_fields_total": len(fields_a),
                "target_fields_total": len(fields_b),
                "results_total": len(results),
            },
        )
        # endregion
        return (
            CompareArtifacts(
                source_pages=list(pages_a),
                target_pages=list(pages_b),
                results=results,
                source_fields=fields_a,
                target_fields=fields_b,
            ),
            pdf_ocr_result_cache,
        )
