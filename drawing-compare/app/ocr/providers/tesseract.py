"""Tesseract OCR via pytesseract."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, cast

from app.models.ocr import OCRDocument, OCRLine, OCRToken
from app.ocr.document_builders import (
    build_document_shell,
    image_size,
    line_id_for,
    make_line_from_tokens,
    read_image_bgr,
)
from app.ocr.document_provider import OCRDependencyError, OCRProvider, OCRProviderError
from app.ocr.geometry_utils import bbox_from_xywh, clamp_confidence


def _require_pytesseract() -> tuple[object, object]:
    try:
        import pytesseract  # noqa: PLC0415
        from pytesseract import Output  # noqa: PLC0415
    except ImportError as e:
        raise OCRDependencyError(
            "pytesseract is not installed. Install optional extra: drawing-compare[ocr-tesseract]."
        ) from e
    return pytesseract, Output


class TesseractOCRProvider(OCRProvider):
    """Word-level OCR using Tesseract; geometry preserved per token and aggregated per line."""

    def __init__(
        self,
        *,
        lang: str = "eng",
        config: str = "",
        psm: int | None = None,
    ) -> None:
        self._lang = lang
        self._config = config
        self._psm = psm

    @property
    def provider_name(self) -> str:
        return "tesseract"

    def extract(self, image_path: str) -> OCRDocument:
        pytesseract, Output = _require_pytesseract()
        output_dict = cast(Any, Output).DICT
        started = time.perf_counter()
        path = self._validate_path(image_path)
        self._log_extract_start(str(path), extra={"lang": self._lang})

        try:
            width_f, height_f = image_size(path)
            bgr = read_image_bgr(path)
            cfg_parts = [self._config] if self._config else []
            if self._psm is not None:
                cfg_parts.append(f"--psm {int(self._psm)}")
            config_str = " ".join(cfg_parts) if cfg_parts else ""

            pt_mod = cast(Any, pytesseract)
            raw = cast(
                dict[str, list[Any]],
                pt_mod.image_to_data(
                    bgr,
                    lang=self._lang,
                    config=config_str or None,
                    output_type=output_dict,
                ),
            )
        except OCRDependencyError:
            raise
        except Exception as e:
            self._log_extract_failure(str(path), e)
            raise OCRProviderError(f"Tesseract failed for {path}: {e}") from e

        n = len(raw.get("text", []))
        line_buckets: dict[tuple[int, int, int], list[tuple[int, OCRToken]]] = defaultdict(list)

        for i in range(n):
            text = (raw.get("text", [""])[i] or "").strip()
            try:
                level = int(raw.get("level", [0])[i])
            except (TypeError, ValueError):
                level = 0
            if level != 5:  # pytesseract WORD level
                continue
            if not text:
                continue
            try:
                conf_raw = int(float(raw.get("conf", [-1])[i]))
            except (TypeError, ValueError):
                conf_raw = -1
            conf = 0.0 if conf_raw < 0 else clamp_confidence(conf_raw, scale_0_100=True)

            left = float(raw.get("left", [0])[i])
            top = float(raw.get("top", [0])[i])
            wi = float(raw.get("width", [0])[i])
            hi = float(raw.get("height", [0])[i])

            block = int(raw.get("block_num", [0])[i])
            par = int(raw.get("par_num", [0])[i])
            line_no = int(raw.get("line_num", [0])[i])
            word_no = int(raw.get("word_num", [0])[i])

            tok = OCRToken(
                text=text,
                confidence=conf,
                bbox=bbox_from_xywh(left, top, wi, hi),
            )
            line_buckets[(block, par, line_no)].append((word_no, tok))

        sorted_line_keys = sorted(line_buckets.keys())
        lines: list[OCRLine] = []
        for idx, lk in enumerate(sorted_line_keys):
            entries = sorted(line_buckets[lk], key=lambda x: x[0])
            tokens = [t for _, t in entries]
            if not tokens:
                continue
            lid = line_id_for(1, idx)
            lines.append(make_line_from_tokens(tokens, line_id=lid, page_no=1, line_idx=idx))

        doc_id = path.stem or path.name or "document"
        meta: dict[str, str | int | float | bool | None] = {
            "tesseract_lang": self._lang,
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
