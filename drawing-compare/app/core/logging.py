"""Structured logging setup for the application."""

import logging
import sys

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure root logger once per process."""
    level = getattr(logging, settings.log_level)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    # Quiet noisy libraries in production
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
