"""Document-level OCR providers and factory wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import fitz
import numpy as np
import pytest

from app.core.config import Settings, reset_settings_cache
from app.models.ocr import BoundingBox, OCRDocument, OCRLine, OCRPage
from app.ocr.document_provider import CloudOCRNotImplementedError, OCRDependencyError
from app.ocr.factory import get_document_ocr_provider, get_ocr_provider
from app.ocr.local_provider import LocalOCRProvider
from app.ocr.providers.windows_ocr import WindowsOCRProvider, winrt_result_to_page
from app.ocr.stub_document_provider import StubDocumentOCRProvider
from app.parsing.fields import RegionParser
from app.preprocessing.settings import PreprocessConfig
from app.reporting.builder import ComparisonReportBuilder
from app.rules.review import ReviewRulesEngine
from app.services.compare import DrawingCompareService


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    img = np.full((32, 48, 3), 255, dtype=np.uint8)
    path = tmp_path / "white.png"
    cv2.imwrite(str(path), img)
    return path


def _build_compare_service(settings: Settings) -> DrawingCompareService:
    rules = ReviewRulesEngine(settings)
    reporter = ComparisonReportBuilder(rules)
    return DrawingCompareService(settings=settings, parser=RegionParser(), reporter=reporter)


def _write_pdf(
    path: Path,
    *,
    width: float = 200.0,
    height: float = 100.0,
    text: str = "PDF",
) -> None:
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_text((24.0, 50.0), text, fontsize=12)
    doc.save(str(path))
    doc.close()


class _PdfOnlyWindowsProvider:
    provider_name = "windows_ocr"

    def __init__(self, template: OCRDocument) -> None:
        self._template = template
        self.calls: list[Path] = []

    def extract(self, image_path: str) -> OCRDocument:
        path = Path(image_path).resolve()
        self.calls.append(path)
        assert path.suffix.lower() == ".pdf"
        return self._template.model_copy(update={"source_path": str(path)})


def test_stub_document_provider_geometry(tiny_png: Path) -> None:
    prov = StubDocumentOCRProvider()
    doc = prov.extract(str(tiny_png))
    assert doc.provider_name == "stub_document"
    assert len(doc.pages) == 1
    assert doc.pages[0].width == 48
    assert doc.pages[0].height == 32
    assert doc.pages[0].lines == []


def test_get_document_ocr_provider_from_settings(tiny_png: Path) -> None:
    reset_settings_cache()
    s = Settings(document_ocr_provider="stub_document")
    prov = get_document_ocr_provider(s)
    assert prov.extract(str(tiny_png)).document_id == "white"


def test_azure_stub_raises(tiny_png: Path) -> None:
    reset_settings_cache()
    s = Settings(document_ocr_provider="azure_document_intelligence")
    prov = get_document_ocr_provider(s)
    with pytest.raises(CloudOCRNotImplementedError):
        prov.extract(str(tiny_png))


def test_get_ndarray_ocr_provider_bridge_stub_document(tiny_png: Path) -> None:
    reset_settings_cache()
    s = Settings(ocr_provider="stub_document")
    prov = get_ocr_provider(s)
    img = cv2.imread(str(tiny_png), cv2.IMREAD_COLOR)
    assert img is not None
    regions = prov.run(img)
    assert regions == []


def test_factory_windows_ocr_and_alias() -> None:
    reset_settings_cache()
    a = get_document_ocr_provider(Settings(document_ocr_provider="windows_ocr"))
    b = get_document_ocr_provider(Settings(document_ocr_provider="windows"))
    assert isinstance(a, WindowsOCRProvider)
    assert isinstance(b, WindowsOCRProvider)
    assert a.provider_name == "windows_ocr"
    assert b.provider_name == "windows_ocr"


def test_windows_ocr_pdf_embedded_text_geometry(tmp_path: Path) -> None:
    pdf_path = tmp_path / "embedded.pdf"
    doc = fitz.open()
    page = doc.new_page(width=320, height=200)
    page.insert_text((48, 96), "Line one for OCR test", fontsize=12)
    page.insert_text((48, 130), "Line two persists", fontsize=12)
    doc.new_page(width=320, height=200)
    doc.save(str(pdf_path))
    doc.close()

    prov = WindowsOCRProvider()
    out = prov.extract(str(pdf_path))
    assert out.provider_name == "windows_ocr"
    assert out.metadata.get("page_1_source") == "embedded_pdf_text"
    assert out.metadata.get("page_2_source") == "embedded_pdf_text"
    assert len(out.pages) == 2
    page1 = out.pages[0]
    assert page1.width == 320
    assert page1.height == 200
    assert len(page1.lines) > 0
    joined = " ".join(line.text for line in page1.lines)
    assert "Line one for OCR test" in joined
    for line in page1.lines:
        assert line.bbox.x2 > line.bbox.x1
        assert line.bbox.y2 > line.bbox.y1
        assert len(line.tokens) > 0
        for tok in line.tokens:
            assert tok.text
            assert tok.bbox.x2 > tok.bbox.x1


def test_winrt_result_to_page_mocked_words() -> None:
    rect = SimpleNamespace(x=10.0, y=20.0, width=30.0, height=12.0)
    word = SimpleNamespace(text="Hello", bounding_rect=rect)
    line = SimpleNamespace(text="Hello", words=[word])
    result = SimpleNamespace(lines=[line])
    lines = winrt_result_to_page(
        result,
        page_number=1,
        width=200.0,
        height=100.0,
        scale_x=2.0,
        scale_y=3.0,
    )
    assert len(lines) == 1
    assert lines[0].text == "Hello"
    assert lines[0].tokens[0].bbox.x1 == 20.0
    assert lines[0].tokens[0].bbox.y1 == 60.0


def test_windows_ocr_raster_raises_when_winrt_missing(tiny_png: Path) -> None:
    prov = WindowsOCRProvider()

    def _raise() -> tuple[object, object, object, object]:
        raise OCRDependencyError("missing winrt")

    with patch("app.ocr.providers.windows_ocr._try_import_winrt_ocr", _raise):
        with pytest.raises(OCRDependencyError):
            prov.extract(str(tiny_png))


def test_compare_service_windows_ocr_uses_direct_pdf_document_path(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source.pdf"
    target_pdf = tmp_path / "target.pdf"
    _write_pdf(source_pdf, text="Source")
    _write_pdf(target_pdf, text="Target")

    template = OCRDocument(
        document_id="doc",
        provider_name="windows_ocr",
        pages=[
            OCRPage(
                page_number=1,
                width=200.0,
                height=100.0,
                lines=[
                    OCRLine(
                        id="p1-l0",
                        text="ALPHA",
                        confidence=0.99,
                        bbox=BoundingBox.from_xywh(50.0, 20.0, 100.0, 20.0),
                    )
                ],
            )
        ],
    )
    provider = _PdfOnlyWindowsProvider(template)
    settings = Settings(
        ocr_provider="windows_ocr",
        preprocess=PreprocessConfig(enable_deskew=False),
    )
    svc = _build_compare_service(settings)

    with patch(
        "app.services.compare.get_ocr_result_provider",
        lambda _s: LocalOCRProvider(inner=provider),
    ):
        with patch(
            "app.services.compare.get_ocr_provider",
            side_effect=AssertionError("raster OCR path should not be used for windows_ocr PDFs"),
        ):
            response, artifacts = svc.compare_paths_with_artifacts(
                source_pdf,
                target_pdf,
                ocr_provider="windows_ocr",
            )

    assert [path.suffix.lower() for path in provider.calls] == [".pdf", ".pdf"]
    assert response.extras["ocr_provider"] == "windows_ocr"
    assert len(artifacts.source_fields) == 1
    source_field = artifacts.source_fields[0]
    assert source_field.value == "ALPHA"
    assert source_field.bbox.x1 == pytest.approx(artifacts.source_page.width * 0.25, rel=0.05)
    assert source_field.bbox.x2 == pytest.approx(artifacts.source_page.width * 0.75, rel=0.05)


def test_compare_service_windows_ocr_multipage_pdf_caches_one_doc_per_path(
    tmp_path: Path,
) -> None:
    """Direct-PDF OCR reuses one OCRDocument per file; each page maps to the correct page_number."""
    source_pdf = tmp_path / "source_2p.pdf"
    target_pdf = tmp_path / "target_2p.pdf"
    for path in (source_pdf, target_pdf):
        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page(width=200, height=100)
            page.insert_text((24.0, 50.0), "x", fontsize=12)
        doc.save(str(path))
        doc.close()

    template = OCRDocument(
        document_id="doc",
        provider_name="windows_ocr",
        pages=[
            OCRPage(
                page_number=1,
                width=200.0,
                height=100.0,
                lines=[
                    OCRLine(
                        id="p1-l0",
                        text="ALPHA",
                        confidence=0.99,
                        bbox=BoundingBox.from_xywh(50.0, 20.0, 100.0, 20.0),
                    )
                ],
            ),
            OCRPage(
                page_number=2,
                width=200.0,
                height=100.0,
                lines=[
                    OCRLine(
                        id="p2-l0",
                        text="BETA",
                        confidence=0.99,
                        bbox=BoundingBox.from_xywh(50.0, 20.0, 100.0, 20.0),
                    )
                ],
            ),
        ],
    )
    provider = _PdfOnlyWindowsProvider(template)
    settings = Settings(
        ocr_provider="windows_ocr",
        preprocess=PreprocessConfig(enable_deskew=False),
    )
    svc = _build_compare_service(settings)

    with patch(
        "app.services.compare.get_ocr_result_provider",
        lambda _s: LocalOCRProvider(inner=provider),
    ):
        with patch(
            "app.services.compare.get_ocr_provider",
            side_effect=AssertionError("raster OCR path should not be used for windows_ocr PDFs"),
        ):
            _response, artifacts = svc.compare_paths_with_artifacts(
                source_pdf,
                target_pdf,
                ocr_provider="windows_ocr",
            )

    assert len(provider.calls) == 2
    assert {str(p.resolve()) for p in provider.calls} == {
        str(source_pdf.resolve()),
        str(target_pdf.resolve()),
    }
    assert len(artifacts.source_fields) == 2
    by_page = {f.page_number: f.value for f in artifacts.source_fields}
    assert by_page[1] == "ALPHA"
    assert by_page[2] == "BETA"
