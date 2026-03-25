"""Shared reviewer-facing output bundle for compare flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.logging import get_logger
from app.models.comparison import CompareResponse
from app.preprocessing.types import BGRImage
from app.reporting.comparison_docx import build_comparison_docx_bytes
from app.reporting.visual_diff import (
    VisualDiffManifest,
    build_visual_diff_html_pages,
    build_visual_diff_manifest,
    image_to_png_bytes,
    render_visual_diff_overlay_pages,
    render_visual_diff_overlay_pdf_pages,
)
from app.services.compare_artifacts import CompareArtifacts

logger = get_logger(__name__)

ReviewerVisualKind = Literal["pdf", "png"]
PDF_VISUAL_FILENAME = "visual_diff_overlay.pdf"
PNG_VISUAL_FILENAME = "visual_diff_overlay.png"
WORD_OUTPUT_FILENAME = "comparison_summary.docx"


@dataclass(frozen=True, slots=True)
class ReviewerBundle:
    """Canonical reviewer-facing bundle for one compare run."""

    manifest: VisualDiffManifest
    comparison_docx: bytes
    overlay_bgr: BGRImage
    visual_bytes: bytes
    visual_kind: ReviewerVisualKind
    html: str | None = None

    @property
    def visual_filename(self) -> str:
        return PDF_VISUAL_FILENAME if self.visual_kind == "pdf" else PNG_VISUAL_FILENAME


def build_reviewer_bundle(
    response: CompareResponse,
    compare_artifacts: CompareArtifacts,
    *,
    include_debug_artifacts: bool = False,
) -> ReviewerBundle:
    """Build the reviewer bundle once, falling back to PNG if PDF output fails."""
    manifest = build_visual_diff_manifest(response, compare_artifacts)
    overlay_bgr = render_visual_diff_overlay_pages(manifest, compare_artifacts)
    html = None
    if include_debug_artifacts:
        html = build_visual_diff_html_pages(manifest, compare_artifacts)

    visual_kind: ReviewerVisualKind = "pdf"
    try:
        visual_bytes = render_visual_diff_overlay_pdf_pages(manifest, compare_artifacts)
    except Exception:
        logger.exception(
            "annotated PDF generation failed comparison_id=%s; falling back to PNG",
            response.comparison_id,
        )
        visual_kind = "png"
        visual_bytes = image_to_png_bytes(overlay_bgr)

    comparison_docx = build_comparison_docx_bytes(response, manifest)
    return ReviewerBundle(
        manifest=manifest,
        comparison_docx=comparison_docx,
        overlay_bgr=overlay_bgr,
        visual_bytes=visual_bytes,
        visual_kind=visual_kind,
        html=html,
    )


def reviewer_output_extras(job_id: str, bundle: ReviewerBundle) -> dict[str, str]:
    """Public reviewer-output contract exposed in response extras."""
    visual_href = f"/compare/artifacts/{job_id}/{bundle.visual_filename}"
    extras: dict[str, str] = {
        "artifact_job_id": job_id,
        "output_visual_href": visual_href,
        "output_visual_kind": bundle.visual_kind,
        "output_word_href": f"/compare/artifacts/{job_id}/{WORD_OUTPUT_FILENAME}",
    }
    if bundle.visual_kind == "pdf":
        extras["output_pdf_href"] = visual_href
    else:
        extras["output_png_href"] = visual_href
    return extras
