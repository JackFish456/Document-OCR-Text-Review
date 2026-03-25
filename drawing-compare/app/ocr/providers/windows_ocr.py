"""Windows-native OCR: embedded PDF text via PyMuPDF, then Windows.Media.Ocr for rasters."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz

from app.models.ocr import BoundingBox, OCRDocument, OCRLine, OCRPage, OCRToken
from app.ocr.document_builders import (
    allocate_token_boxes_along_line,
    image_size,
    line_id_for,
    make_line_from_tokens,
)
from app.ocr.document_provider import (
    OCRDependencyError,
    OCRImageLoadError,
    OCRProvider,
    OCRProviderError,
)
from app.ocr.geometry_utils import bbox_from_xywh, clamp_confidence

_PDF_SUFFIXES = {".pdf"}
_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_DEFAULT_PDF_OCR_DPI = 200
_WINRT_OCR_INSTALL_HINT = (
    "Install the optional Windows OCR bindings: pip install 'drawing-compare[ocr-windows]' "
    "(or manually: winrt-Windows.Media.Ocr, winrt-Windows.Graphics.Imaging, "
    "winrt-Windows.Storage.Streams)."
)


def _run_coroutine(coro: Any) -> Any:
    """Run an async WinRT coroutine from sync code (handles nested event loops)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def _in_thread() -> Any:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


def _bounding_rect_to_box(rect: Any) -> BoundingBox:
    x = float(getattr(rect, "x", getattr(rect, "X", 0.0)))
    y = float(getattr(rect, "y", getattr(rect, "Y", 0.0)))
    w = float(getattr(rect, "width", getattr(rect, "Width", 0.0)))
    h = float(getattr(rect, "height", getattr(rect, "Height", 0.0)))
    return bbox_from_xywh(x, y, w, h)


def _scale_box(box: BoundingBox, *, scale_x: float, scale_y: float) -> BoundingBox:
    return bbox_from_xywh(
        box.x1 * scale_x,
        box.y1 * scale_y,
        box.width() * scale_x,
        box.height() * scale_y,
    )


def _try_import_winrt_ocr() -> tuple[Any, Any, Any, Any]:
    try:
        from winrt.windows.graphics.imaging import BitmapDecoder  # noqa: PLC0415
        from winrt.windows.media.ocr import OcrEngine  # noqa: PLC0415
        from winrt.windows.storage.streams import (  # noqa: PLC0415
            DataWriter,
            InMemoryRandomAccessStream,
        )
    except ImportError as e:
        raise OCRDependencyError(
            f"WinRT OCR bindings are not available. {_WINRT_OCR_INSTALL_HINT}"
        ) from e
    return (
        BitmapDecoder.create_async,
        OcrEngine,
        DataWriter,
        InMemoryRandomAccessStream,
    )


async def _winrt_recognize_image_bytes_async(image_bytes: bytes) -> tuple[Any, int, int]:
    create_decoder_async, ocr_engine_cls, data_writer_cls, stream_cls = _try_import_winrt_ocr()
    stream = stream_cls()
    writer = data_writer_cls(stream)
    writer.write_bytes(image_bytes)
    await writer.store_async()
    writer.detach_stream()
    stream.seek(0)
    decoder = await create_decoder_async(stream)
    software_bitmap = await decoder.get_software_bitmap_async()
    pixel_width = int(software_bitmap.pixel_width)
    pixel_height = int(software_bitmap.pixel_height)
    engine = ocr_engine_cls.try_create_from_user_profile_languages()
    if engine is None:
        raise OCRProviderError(
            "Windows OCR engine could not be created from user profile languages. "
            "Install at least one Windows OCR language pack under Settings > Time & language > "
            "Language & region > Preferred languages (add a language, then Optional features > "
            "Optical Character Recognition)."
        )
    result = await engine.recognize_async(software_bitmap)
    return result, pixel_width, pixel_height


def winrt_result_to_page(
    result: Any,
    *,
    page_number: int,
    width: float,
    height: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> list[OCRLine]:
    """Convert WinRT OcrResult lines; optional scale maps bitmap coords to page space."""
    lines_out: list[OCRLine] = []
    raw_lines = list(getattr(result, "lines", []) or [])
    for idx, line in enumerate(raw_lines):
        lid = line_id_for(page_number, idx)
        words = list(getattr(line, "words", []) or [])
        if words:
            tokens: list[OCRToken] = []
            for word in words:
                text = (getattr(word, "text", None) or "").strip()
                if not text:
                    continue
                word_box = _scale_box(
                    _bounding_rect_to_box(word.bounding_rect),
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                tokens.append(
                    OCRToken(
                        text=text,
                        confidence=clamp_confidence(0.95),
                        bbox=word_box,
                        line_id=lid,
                    )
                )
            if tokens:
                lines_out.append(
                    make_line_from_tokens(tokens, line_id=lid, page_no=page_number, line_idx=idx)
                )
            continue

        line_text = (getattr(line, "text", None) or "").strip()
        if not line_text:
            continue
        raw_line_box = getattr(line, "bounding_rect", None)
        if raw_line_box is None:
            line_box = bbox_from_xywh(0.0, 0.0, width, max(height * 0.15, 1.0))
        else:
            line_box = _scale_box(
                _bounding_rect_to_box(raw_line_box),
                scale_x=scale_x,
                scale_y=scale_y,
            )
        tokens = allocate_token_boxes_along_line(
            line_text,
            line_box,
            base_confidence=0.9,
            line_id=lid,
        )
        if tokens:
            lines_out.append(
                make_line_from_tokens(tokens, line_id=lid, page_no=page_number, line_idx=idx)
            )
    return lines_out


def _assemble_multipage_document(
    *,
    document_id: str,
    source_path: str,
    provider_name: str,
    pages: list[OCRPage],
    metadata: dict[str, str | int | float | bool | None],
) -> OCRDocument:
    stamped: list[OCRPage] = []
    for page in pages:
        stamped_lines = [line.with_token_line_ids() for line in page.lines]
        stamped.append(
            OCRPage(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                lines=stamped_lines,
            )
        )
    meta = dict(metadata)
    meta["line_count"] = sum(len(page.lines) for page in stamped)
    return OCRDocument(
        document_id=document_id,
        source_path=source_path,
        provider_name=provider_name,
        pages=stamped,
        metadata=meta,
    )


def _lines_from_pymupdf_words(page: Any, *, page_no: int) -> list[OCRLine]:
    grouped: dict[tuple[int, int], list[tuple[int, OCRToken]]] = defaultdict(list)
    words = list(page.get_text("words", sort=True) or [])
    for raw_word in words:
        if len(raw_word) < 5:
            continue
        x0, y0, x1, y1 = (
            float(raw_word[0]),
            float(raw_word[1]),
            float(raw_word[2]),
            float(raw_word[3]),
        )
        text = str(raw_word[4]).strip()
        if not text:
            continue
        block_no = int(raw_word[5]) if len(raw_word) > 5 else 0
        line_no = int(raw_word[6]) if len(raw_word) > 6 else 0
        word_no = int(raw_word[7]) if len(raw_word) > 7 else len(grouped[(block_no, line_no)])
        grouped[(block_no, line_no)].append(
            (
                word_no,
                OCRToken(
                    text=text,
                    confidence=clamp_confidence(1.0),
                    bbox=bbox_from_xywh(x0, y0, x1 - x0, y1 - y0),
                ),
            )
        )

    lines: list[OCRLine] = []
    for idx, key in enumerate(sorted(grouped)):
        entries = sorted(grouped[key], key=lambda item: item[0])
        tokens = [token for _, token in entries]
        if not tokens:
            continue
        line_id = line_id_for(page_no, idx)
        lines.append(make_line_from_tokens(tokens, line_id=line_id, page_no=page_no, line_idx=idx))
    return lines


def _line_from_pymupdf_line(
    line_data: dict[str, Any],
    page_no: int,
    line_idx: int,
) -> OCRLine | None:
    spans_raw = line_data.get("spans") or []
    tokens: list[OCRToken] = []
    lid = line_id_for(page_no, line_idx)
    for span in spans_raw:
        text = (span.get("text") or "").strip()
        if not text:
            continue
        bbox = span.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        tokens.append(
            OCRToken(
                text=text,
                confidence=clamp_confidence(1.0),
                bbox=bbox_from_xywh(x0, y0, x1 - x0, y1 - y0),
                line_id=lid,
            )
        )
    if not tokens:
        return None
    return make_line_from_tokens(tokens, line_id=lid, page_no=page_no, line_idx=line_idx)


def _extract_pdf_embedded_text(path: Path) -> OCRDocument | None:
    pdf = fitz.open(str(path))
    try:
        pages_out: list[OCRPage] = []
        total_lines = 0
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_no = page_index + 1
            lines = _lines_from_pymupdf_words(page, page_no=page_no)
            if not lines:
                text_dict = page.get_text("dict")
                lines = []
                line_idx = 0
                for block in text_dict.get("blocks") or []:
                    if block.get("type") != 0:
                        continue
                    for line_info in block.get("lines") or []:
                        line = _line_from_pymupdf_line(line_info, page_no, line_idx)
                        if line is None:
                            continue
                        lines.append(line)
                        line_idx += 1
            total_lines += len(lines)
            pages_out.append(
                OCRPage(
                    page_number=page_no,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    lines=lines,
                )
            )
    finally:
        pdf.close()

    if total_lines == 0:
        return None

    doc_id = path.stem or path.name or "document"
    metadata: dict[str, str | int | float | bool | None] = {
        "extraction": "embedded_pdf_text",
        "page_count": len(pages_out),
    }
    for page in pages_out:
        metadata[f"page_{page.page_number}_source"] = "embedded_pdf_text"
    return _assemble_multipage_document(
        document_id=doc_id,
        source_path=str(path),
        provider_name="windows_ocr",
        pages=pages_out,
        metadata=metadata,
    )


def _extract_pdf_via_windows_ocr(path: Path, *, dpi: int = _DEFAULT_PDF_OCR_DPI) -> OCRDocument:
    pdf = fitz.open(str(path))
    pages_out: list[OCRPage] = []
    metadata: dict[str, str | int | float | bool | None] = {"extraction": "windows_ocr"}
    try:
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_no = page_index + 1
            pdf_w = float(page.rect.width)
            pdf_h = float(page.rect.height)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            result, pix_w, pix_h = _run_coroutine(
                _winrt_recognize_image_bytes_async(pix.tobytes("png"))
            )
            lines = winrt_result_to_page(
                result,
                page_number=page_no,
                width=pdf_w,
                height=pdf_h,
                scale_x=pdf_w / float(pix_w),
                scale_y=pdf_h / float(pix_h),
            )
            pages_out.append(
                OCRPage(
                    page_number=page_no,
                    width=pdf_w,
                    height=pdf_h,
                    lines=lines,
                )
            )
            metadata[f"page_{page_no}_source"] = "windows_ocr"
    finally:
        pdf.close()

    doc_id = path.stem or path.name or "document"
    metadata["page_count"] = len(pages_out)
    return _assemble_multipage_document(
        document_id=doc_id,
        source_path=str(path),
        provider_name="windows_ocr",
        pages=pages_out,
        metadata=metadata,
    )


def _extract_with_windows_ocr(path: Path) -> OCRDocument:
    width_f, height_f = image_size(path)
    try:
        image_bytes = path.read_bytes()
    except OSError as e:
        raise OCRImageLoadError(f"Could not read raster image bytes: {path}") from e
    result, _, _ = _run_coroutine(_winrt_recognize_image_bytes_async(image_bytes))
    lines = winrt_result_to_page(
        result,
        page_number=1,
        width=float(width_f),
        height=float(height_f),
    )
    doc_id = path.stem or path.name or "document"
    meta: dict[str, str | int | float | bool | None] = {
        "extraction": "windows_ocr",
        "page_count": 1,
        "page_1_source": "windows_ocr",
    }
    return _assemble_multipage_document(
        document_id=doc_id,
        source_path=str(path),
        provider_name="windows_ocr",
        pages=[OCRPage(page_number=1, width=float(width_f), height=float(height_f), lines=lines)],
        metadata=meta,
    )


class WindowsOCRProvider(OCRProvider):
    """Hybrid PDF (embedded text, then raster+WinRT) and raster image OCR for Windows."""

    def __init__(self, *, pdf_raster_dpi: int = _DEFAULT_PDF_OCR_DPI) -> None:
        self._pdf_raster_dpi = pdf_raster_dpi

    @property
    def provider_name(self) -> str:
        return "windows_ocr"

    def extract(self, image_path: str) -> OCRDocument:
        started = time.perf_counter()
        path = self._validate_path(image_path)
        suffix = path.suffix.lower()

        if suffix not in _PDF_SUFFIXES | _RASTER_SUFFIXES:
            raise OCRImageLoadError(
                f"Unsupported file type for windows_ocr: {suffix!r}. "
                f"Expected a PDF or raster image {_RASTER_SUFFIXES}."
            )

        self._log_extract_start(str(path), extra={"suffix": suffix})

        try:
            if suffix in _PDF_SUFFIXES:
                embedded = _extract_pdf_embedded_text(path)
                if embedded is not None:
                    doc = embedded
                else:
                    doc = _extract_pdf_via_windows_ocr(path, dpi=self._pdf_raster_dpi)
            else:
                doc = _extract_with_windows_ocr(path)
        except (OCRDependencyError, OCRImageLoadError, OCRProviderError):
            raise
        except Exception as e:
            self._log_extract_failure(str(path), e)
            raise OCRProviderError(f"Windows OCR failed for {path}: {e}") from e

        elapsed = time.perf_counter() - started
        self._log_extract_done(
            str(path),
            pages=len(doc.pages),
            lines=sum(len(p.lines) for p in doc.pages),
            tokens=len(doc.all_tokens_flat()),
            duration_s=round(elapsed, 4),
        )
        return doc
