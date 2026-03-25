"""Rule-based field extraction from OCR lines."""

from app.parsing.line_fields.base import LineFieldExtractor
from app.parsing.line_fields.context import LineExtractionContext
from app.parsing.line_fields.engine import (
    extract_fields_from_document,
    extract_fields_from_line,
    extract_fields_from_page,
)
from app.parsing.line_fields.matches import ExtractionMatch
from app.parsing.line_fields.regex_extractor import RegexLineExtractor
from app.parsing.line_fields.rules import default_line_extractors

__all__ = [
    "ExtractionMatch",
    "LineExtractionContext",
    "LineFieldExtractor",
    "RegexLineExtractor",
    "default_line_extractors",
    "extract_fields_from_document",
    "extract_fields_from_line",
    "extract_fields_from_page",
]
