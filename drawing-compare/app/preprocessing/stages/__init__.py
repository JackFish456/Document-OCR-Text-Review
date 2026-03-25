"""Individual preprocessing stages (pure functions where possible)."""

from app.preprocessing.stages.contrast import apply_clahe
from app.preprocessing.stages.denoise import denoise_gray
from app.preprocessing.stages.grayscale import to_gray
from app.preprocessing.stages.threshold import adaptive_threshold
from app.preprocessing.stages.tiling import build_tiles, should_tile

__all__ = [
    "adaptive_threshold",
    "apply_clahe",
    "build_tiles",
    "denoise_gray",
    "should_tile",
    "to_gray",
]
