"""Re-export of :mod:`app.ocr.ocr_comparator` (implementation lives under ``app.ocr`` for import hygiene)."""

from app.ocr.ocr_comparator import OCRComparisonMetrics, compare_ocr

__all__ = ["OCRComparisonMetrics", "compare_ocr"]
