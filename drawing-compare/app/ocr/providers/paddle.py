"""PaddleOCR line-level backend with token geometry derived along each line."""

from __future__ import annotations

import time
from typing import Any

from app.models.ocr import OCRDocument, OCRLine
from app.ocr.document_builders import (
    allocate_token_boxes_along_line,
    build_document_shell,
    image_size,
    line_id_for,
    make_line_from_tokens,
)
from app.ocr.document_provider import OCRDependencyError, OCRProvider, OCRProviderError
from app.ocr.geometry_utils import bbox_from_quad, clamp_confidence


def _require_paddle() -> Any:
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
    except ImportError as e:
        raise OCRDependencyError(
            "paddleocr is not installed. Install optional extra: drawing-compare[ocr-paddle]."
        ) from e
    return PaddleOCR


class PaddleOCRProvider(OCRProvider):
    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        show_log: bool = False,
    ) -> None:
        self._lang = lang
        self._use_angle_cls = use_angle_cls
        self._use_gpu = use_gpu
        self._show_log = show_log
        self._engine: Any | None = None

    @property
    def provider_name(self) -> str:
        return "paddleocr"

    def _get_engine(self) -> Any:
        if self._engine is None:
            paddle_ocr_cls = _require_paddle()
            self._engine = paddle_ocr_cls(
                use_angle_cls=self._use_angle_cls,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=self._show_log,
            )
        return self._engine

    def extract(self, image_path: str) -> OCRDocument:
        _require_paddle()
        started = time.perf_counter()
        path = self._validate_path(image_path)
        self._log_extract_start(str(path), extra={"lang": self._lang})

        try:
            width_f, height_f = image_size(path)
            engine = self._get_engine()
            raw: Any = engine.ocr(str(path), cls=self._use_angle_cls)
        except OCRDependencyError:
            raise
        except Exception as e:
            self._log_extract_failure(str(path), e)
            raise OCRProviderError(f"PaddleOCR failed for {path}: {e}") from e

        page_blocks = raw[0] if raw and len(raw) > 0 else None
        lines: list[OCRLine] = []
        if page_blocks:
            for idx, item in enumerate(page_blocks):
                if not item or len(item) < 2:
                    continue
                box_pts, tx = item[0], item[1]
                if not tx or len(tx) < 2:
                    continue
                text, score = str(tx[0]), float(tx[1])
                pts = [(float(p[0]), float(p[1])) for p in box_pts]
                line_box = bbox_from_quad(pts)
                lid = line_id_for(1, idx)
                toks = allocate_token_boxes_along_line(
                    text,
                    line_box,
                    base_confidence=clamp_confidence(score),
                    line_id=lid,
                )
                if not toks:
                    continue
                lines.append(make_line_from_tokens(toks, line_id=lid, page_no=1, line_idx=idx))

        doc_id = path.stem or path.name or "document"
        meta: dict[str, str | int | float | bool | None] = {
            "paddle_lang": self._lang,
            "line_count": len(lines),
        }
        doc = build_document_shell(
            document_id=doc_id,
            source_path=str(path),
            provider_name=self.provider_name,
            width=float(width_f),
            height=float(height_f),
            lines=lines,
            metadata=meta,
        )
        elapsed = time.perf_counter() - started
        self._log_extract_done(
            str(path),
            pages=len(doc.pages),
            lines=len(lines),
            tokens=len(doc.all_tokens_flat()),
            duration_s=round(elapsed, 4),
        )
        return doc
