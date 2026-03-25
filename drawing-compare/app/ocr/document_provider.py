"""File-oriented OCR provider abstraction and shared errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.logging import get_logger
from app.models.ocr import OCRDocument

logger = get_logger(__name__)


class OCRError(Exception):
    """Base class for OCR pipeline failures."""


class OCRDependencyError(OCRError):
    """Raised when an optional backend or system dependency is missing."""


class OCRImageLoadError(OCRError):
    """Raised when the image path cannot be read or decoded."""


class OCRProviderError(OCRError):
    """Raised when the provider fails during extraction."""


class CloudOCRNotImplementedError(OCRError):
    """Raised by stub cloud providers until SDK credentials are wired."""


class OCRProvider(ABC):
    """File-based OCR: read an image path and return a normalized ``OCRDocument``."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable identifier for logging, serialization, and config."""

    @abstractmethod
    def extract(self, image_path: str) -> OCRDocument:
        """Run OCR on ``image_path`` and return a normalized document."""

    def _validate_path(self, image_path: str) -> Path:
        p = Path(image_path).expanduser().resolve()
        if not p.is_file():
            raise OCRImageLoadError(f"Not a file or missing: {image_path!r}")
        return p

    def _log_extract_start(self, image_path: str, extra: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = {"provider": self.provider_name, "path": image_path}
        if extra:
            payload.update(extra)
        logger.info("OCR extract started", extra=payload)

    def _log_extract_done(
        self,
        image_path: str,
        *,
        pages: int,
        lines: int,
        tokens: int,
        duration_s: float | None = None,
    ) -> None:
        logger.info(
            "OCR extract finished",
            extra={
                "provider": self.provider_name,
                "path": image_path,
                "pages": pages,
                "lines": lines,
                "tokens": tokens,
                "duration_s": duration_s,
            },
        )

    def _log_extract_failure(self, image_path: str, exc: BaseException) -> None:
        logger.exception(
            "OCR extract failed",
            extra={"provider": self.provider_name, "path": image_path, "error": str(exc)},
        )
