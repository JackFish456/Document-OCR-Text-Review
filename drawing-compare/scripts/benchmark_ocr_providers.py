#!/usr/bin/env python3
"""Run multiple file-based OCR providers on the same image and compare coverage vs. confidence.

Usage (from repo root ``drawing-compare/``)::

    python scripts/benchmark_ocr_providers.py path/to/image.png --out-dir ./ocr_benchmark_out

Environment/config follows :class:`app.core.config.Settings` (``DRAWING_COMPARE_*``).
CLI ``--provider`` flags override which backends are exercised.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow ``python scripts/...`` without editable install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import Settings  # noqa: E402
from app.models.ocr import OCRDocument, normalize_ocr_text  # noqa: E402
from app.models.serialization import model_to_json  # noqa: E402
from app.ocr.document_provider import (  # noqa: E402
    CloudOCRNotImplementedError,
    OCRError,
    OCRProvider,
)
from app.ocr.factory import get_document_ocr_provider  # noqa: E402
from app.ocr.providers.azure import AzureDocumentIntelligenceProvider  # noqa: E402
from app.ocr.providers.google_vision import GoogleVisionOCRProvider  # noqa: E402
from app.ocr.providers.paddle import PaddleOCRProvider  # noqa: E402
from app.ocr.providers.tesseract import TesseractOCRProvider  # noqa: E402
from app.ocr.stub_document_provider import StubDocumentOCRProvider  # noqa: E402


@dataclass
class BenchmarkRow:
    provider: str
    ok: bool
    error: str | None
    duration_s: float
    line_count: int
    token_count: int
    text_chars: int
    normalized_chars: int
    mean_token_confidence: float | None
    min_token_confidence: float | None


def _build_provider_by_name(name: str, settings: Settings) -> OCRProvider:
    key = name.lower().strip()
    if key in ("stub", "stub_document"):
        return StubDocumentOCRProvider()
    if key == "tesseract":
        return TesseractOCRProvider(
            lang=settings.tesseract_lang,
            config=settings.tesseract_config,
            psm=settings.tesseract_psm,
        )
    if key in ("paddle", "paddleocr"):
        return PaddleOCRProvider(
            lang=settings.paddle_lang,
            use_angle_cls=settings.paddle_use_angle_cls,
            use_gpu=settings.paddle_use_gpu,
        )
    if key in ("azure", "azure_document_intelligence"):
        return AzureDocumentIntelligenceProvider()
    if key in ("google", "google_vision"):
        return GoogleVisionOCRProvider()
    return get_document_ocr_provider(
        settings.model_copy(update={"document_ocr_provider": name})  # type: ignore[call-arg]
    )


def _metrics_from_document(
    doc: OCRDocument,
) -> tuple[int, int, int, int, float | None, float | None]:
    tokens = doc.all_tokens_flat()
    lines = doc.all_lines_flat()
    joined = " ".join(t.text for t in tokens)
    norm = normalize_ocr_text(joined)
    confs = [t.confidence for t in tokens]
    mean_c = sum(confs) / len(confs) if confs else None
    min_c = min(confs) if confs else None
    return len(lines), len(tokens), len(joined), len(norm), mean_c, min_c


def run_benchmark(
    image_path: Path,
    *,
    providers: list[str],
    out_dir: Path,
    settings: Settings,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkRow] = []
    artifacts: dict[str, str | None] = {}

    for pname in providers:
        prov = _build_provider_by_name(pname, settings)
        t0 = time.perf_counter()
        error: str | None = None
        doc: OCRDocument | None = None
        try:
            doc = prov.extract(str(image_path.resolve()))
        except CloudOCRNotImplementedError as e:
            error = f"not_implemented: {e}"
        except OCRError as e:
            error = f"ocr_error: {e}"
        except Exception as e:  # noqa: BLE001 — benchmark harness
            error = f"unexpected: {type(e).__name__}: {e}"
        duration = time.perf_counter() - t0

        json_name = f"{out_dir / image_path.stem}_{prov.provider_name}.json"
        if doc is not None:
            json_name_path = Path(json_name)
            json_name_path.write_text(
                model_to_json(doc, indent=2, exclude_none=True),
                encoding="utf-8",
            )
            artifacts[prov.provider_name] = str(json_name_path.resolve())
            lc, tc, ch, nch, mean_c, min_c = _metrics_from_document(doc)
            rows.append(
                BenchmarkRow(
                    provider=prov.provider_name,
                    ok=True,
                    error=None,
                    duration_s=round(duration, 4),
                    line_count=lc,
                    token_count=tc,
                    text_chars=ch,
                    normalized_chars=nch,
                    mean_token_confidence=mean_c,
                    min_token_confidence=min_c,
                )
            )
        else:
            artifacts[prov.provider_name] = None
            rows.append(
                BenchmarkRow(
                    provider=prov.provider_name,
                    ok=False,
                    error=error,
                    duration_s=round(duration, 4),
                    line_count=0,
                    token_count=0,
                    text_chars=0,
                    normalized_chars=0,
                    mean_token_confidence=None,
                    min_token_confidence=None,
                )
            )

    # Comparison: rank successful runs by normalized char coverage, then mean confidence
    ok_rows = [r for r in rows if r.ok and r.token_count > 0]
    ranking = sorted(
        ok_rows,
        key=lambda r: (r.normalized_chars, r.mean_token_confidence or 0.0),
        reverse=True,
    )
    summary: dict[str, object] = {
        "image": str(image_path.resolve()),
        "providers_requested": providers,
        "artifacts": artifacts,
        "rows": [r.__dict__ for r in rows],
        "ranking_by_coverage_then_confidence": [r.provider for r in ranking],
    }
    summary_path = out_dir / f"{image_path.stem}_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to image (png/jpg/tiff/…)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ocr_benchmark_out"),
        help="Directory for per-provider JSON and summary",
    )
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        default=[],
        help=(
            "Provider key (repeatable). Examples: stub_document, tesseract, "
            "paddleocr, windows_ocr"
        ),
    )
    args = parser.parse_args()
    settings = Settings()

    providers = args.providers or ["stub_document", "tesseract", "paddleocr"]
    run_benchmark(args.image, providers=providers, out_dir=args.out_dir, settings=settings)

    print(f"Wrote benchmark artifacts under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
