"""Document-level stub provider (empty text, valid page geometry)."""

from __future__ import annotations

import time

from app.models.ocr import OCRDocument
from app.ocr.document_builders import build_document_shell, image_size
from app.ocr.document_provider import OCRProvider


class StubDocumentOCRProvider(OCRProvider):
    """Returns a single page with no lines/tokens — for tests and offline API wiring."""

    @property
    def provider_name(self) -> str:
        return "stub_document"

    def extract(self, image_path: str) -> OCRDocument:
        started = time.perf_counter()
        path = self._validate_path(image_path)
        self._log_extract_start(str(path))
        w, h = image_size(path)
        doc_id = path.stem or path.name or "document"
        doc = build_document_shell(
            document_id=doc_id,
            source_path=str(path),
            provider_name=self.provider_name,
            width=float(w),
            height=float(h),
            lines=[],
            metadata={"stub": True},
        )
        elapsed = time.perf_counter() - started
        self._log_extract_done(
            str(path),
            pages=1,
            lines=0,
            tokens=0,
            duration_s=round(elapsed, 4),
        )
        return doc
