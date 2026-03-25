"""FastAPI dependency providers."""

from functools import lru_cache

from app.core.config import get_settings
from app.parsing.fields import RegionParser
from app.reporting.builder import ComparisonReportBuilder
from app.reporting.llm_summary import summarizer_from_settings
from app.rules.review import ReviewRulesEngine
from app.services.compare import DrawingCompareService


def build_compare_service() -> DrawingCompareService:
    settings = get_settings()
    parser = RegionParser()
    rules = ReviewRulesEngine(settings)
    reporter = ComparisonReportBuilder(
        rules,
        llm_summarizer=summarizer_from_settings(settings.report_llm_summary),
    )
    return DrawingCompareService(
        settings=settings,
        parser=parser,
        reporter=reporter,
    )


@lru_cache(maxsize=1)
def get_compare_service() -> DrawingCompareService:
    """Single compare service instance per process (settings loaded once)."""
    return build_compare_service()
