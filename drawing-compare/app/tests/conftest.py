"""Pytest configuration: repo-local temp roots for Windows / OneDrive reliability.

Pytest's default temp hierarchy lives under the OS temp directory (e.g.
``%LOCALAPPDATA%\\Temp\\pytest-of-<user>``), which can fail with permission or
sync issues on some Windows setups. Redirecting the temp *root* to a path under
the repository keeps ``tmp_path`` / ``tmp_path_factory`` on stable, writable
storage without using ``--basetemp`` (which deletes the entire basetemp if it
exists at session start).
"""

from __future__ import annotations

import os
from pathlib import Path

# app/tests/conftest.py -> parents[2] == project root (drawing-compare/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTEST_TEMPROOT = _REPO_ROOT / "data" / "tmp" / "pytest"


def _ensure_pytest_temproot() -> None:
    """Set ``PYTEST_DEBUG_TEMPROOT`` so pytest builds ``pytest-of-*`` under the repo."""
    _PYTEST_TEMPROOT.mkdir(parents=True, exist_ok=True)
    # Let an explicit environment override win (e.g. CI or local debugging).
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_PYTEST_TEMPROOT.resolve()))


# Apply as soon as this conftest is imported (during collection for tests under app/tests).
_ensure_pytest_temproot()
