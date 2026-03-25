# ruff: noqa: E501
"""Visual overlay diff artifacts for local drawing review."""

from __future__ import annotations

import base64
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from pydantic import BaseModel, Field, model_validator

from app.models.comparison import CompareResponse
from app.models.match import ComparisonSummary, MatchResult, MatchType
from app.models.ocr import BoundingBox
from app.preprocessing.types import BGRImage, PreprocessPageResult
from app.reporting.display import field_caption
from app.services.compare_artifacts import CompareArtifacts

_VISIBLE_MATCH_TYPES = frozenset(
    {
        MatchType.CHANGED_VALUE,
        MatchType.MISSING_IN_TARGET,
        MatchType.EXTRA_IN_TARGET,
        MatchType.UNCERTAIN,
    }
)
_TARGET_CROP_MATCH_TYPES = frozenset(
    {
        MatchType.CHANGED_VALUE,
        MatchType.EXTRA_IN_TARGET,
        MatchType.UNCERTAIN,
    }
)


class NormalizedBoundingBox(BaseModel):
    """Bounding box normalized to page width and height."""

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _valid_extent(self) -> NormalizedBoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("NormalizedBoundingBox requires x2 > x1 and y2 > y1")
        return self

    @classmethod
    def from_bbox(
        cls,
        bbox: BoundingBox,
        *,
        width: int,
        height: int,
    ) -> NormalizedBoundingBox:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        return cls(
            x1=_round_unit(bbox.x1 / float(width)),
            y1=_round_unit(bbox.y1 / float(height)),
            x2=_round_unit(bbox.x2 / float(width)),
            y2=_round_unit(bbox.y2 / float(height)),
        )

    @classmethod
    def from_pixel_bounds(
        cls,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
    ) -> NormalizedBoundingBox:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        return cls(
            x1=_round_unit(x1 / float(width)),
            y1=_round_unit(y1 / float(height)),
            x2=_round_unit(x2 / float(width)),
            y2=_round_unit(y2 / float(height)),
        )


class VisualAnnotation(BaseModel):
    """Stable visual record for one non-exact comparison result."""

    annotation_id: str = Field(min_length=1)
    index: int = Field(ge=1)
    match_type: MatchType
    source_field_id: str | None = None
    target_field_id: str | None = None
    source_text: str = ""
    target_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None
    source_bbox: BoundingBox | None = None
    target_bbox: BoundingBox | None = None
    source_bbox_normalized: NormalizedBoundingBox | None = None
    target_bbox_normalized: NormalizedBoundingBox | None = None
    overlay_bbox_normalized: NormalizedBoundingBox


class VisualDiffPageSlice(BaseModel):
    """One aligned index in a multi-page compare (source page N vs target page N)."""

    pair_index: int = Field(ge=0, description="0-based index into the page-aligned compare")
    source_page_number: int = Field(default=0, ge=0, description="0 if no source page in this slice")
    target_page_number: int = Field(default=0, ge=0, description="0 if no target page in this slice")
    source_image_width: int = Field(ge=0)
    source_image_height: int = Field(ge=0)
    target_image_width: int = Field(ge=0)
    target_image_height: int = Field(ge=0)
    annotations: list[VisualAnnotation] = Field(default_factory=list)


class VisualDiffManifest(BaseModel):
    """JSON-serializable manifest for visual diff artifacts."""

    schema_version: str = "2.0"
    comparison_id: str = Field(min_length=1)
    source_path: str = ""
    target_path: str = ""
    ocr_provider: str = ""
    source_page_number: int = Field(default=1, ge=1)
    target_page_number: int = Field(default=1, ge=1)
    source_image_width: int = Field(gt=0)
    source_image_height: int = Field(gt=0)
    target_image_width: int = Field(gt=0)
    target_image_height: int = Field(gt=0)
    summary: ComparisonSummary
    annotation_count: int = Field(default=0, ge=0)
    annotations: list[VisualAnnotation] = Field(default_factory=list)
    pages: list[VisualDiffPageSlice] = Field(
        default_factory=list,
        description="Per-page slices; flattened annotations duplicate this for convenience",
    )

    @model_validator(mode="after")
    def _sync_annotation_count(self) -> VisualDiffManifest:
        self.annotation_count = len(self.annotations)
        return self


@dataclass(slots=True)
class VisualDiffArtifacts:
    """Rendered outputs for local/manual review runs."""

    manifest: VisualDiffManifest
    overlay_pdf: bytes
    overlay_bgr: BGRImage
    html: str


@dataclass(frozen=True, slots=True)
class BadgePlacement:
    """Resolved badge position for one annotation."""

    annotation_id: str
    index: int
    placement: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def normalized_bbox(
        self,
        *,
        page_width: int,
        page_height: int,
    ) -> NormalizedBoundingBox:
        return NormalizedBoundingBox.from_pixel_bounds(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            width=page_width,
            height=page_height,
        )


@dataclass(frozen=True, slots=True)
class _VisualStyle:
    label: str
    color_hex: str
    fill_alpha: float
    crop_alpha: float
    pdf_fill_alpha: float

    @property
    def rgb(self) -> tuple[int, int, int]:
        return _hex_to_rgb(self.color_hex)

    @property
    def bgr(self) -> tuple[int, int, int]:
        r, g, b = self.rgb
        return b, g, r

    @property
    def pdf_rgb(self) -> tuple[float, float, float]:
        r, g, b = self.rgb
        return r / 255.0, g / 255.0, b / 255.0


_STYLE_BY_MATCH_TYPE: dict[MatchType, _VisualStyle] = {
    MatchType.CHANGED_VALUE: _VisualStyle("Changed text", "#d97706", 0.12, 0.48, 0.08),
    MatchType.MISSING_IN_TARGET: _VisualStyle("Missing on target", "#b42318", 0.26, 0.0, 0.14),
    MatchType.EXTRA_IN_TARGET: _VisualStyle("Only on target", "#0f766e", 0.12, 0.44, 0.08),
    MatchType.UNCERTAIN: _VisualStyle("Unclear match", "#7c6f10", 0.18, 0.34, 0.10),
}


def _belongs_to_pair_slice(
    result: MatchResult,
    *,
    sp: PreprocessPageResult | None,
    tp: PreprocessPageResult | None,
) -> bool:
    """Match results are grouped by aligned page index (same page_number on each side when both exist)."""
    if result.match_type == MatchType.EXTRA_IN_TARGET:
        tf = result.target_field
        if tf is None or tp is None:
            return False
        return tf.page_number == tp.page_number
    if result.match_type == MatchType.MISSING_IN_TARGET:
        sf = result.source_field
        if sf is None or sp is None:
            return False
        return sf.page_number == sp.page_number
    sf, tf = result.source_field, result.target_field
    if sf is None or tf is None:
        return False
    ok_s = sp is None or sf.page_number == sp.page_number
    ok_t = tp is None or tf.page_number == tp.page_number
    return ok_s and ok_t


def _pair_index_for_result(
    result: MatchResult,
    *,
    src_pages: list[PreprocessPageResult],
    tgt_pages: list[PreprocessPageResult],
) -> int:
    n_pairs = max(len(src_pages), len(tgt_pages))
    for pair_index in range(n_pairs):
        sp = src_pages[pair_index] if pair_index < len(src_pages) else None
        tp = tgt_pages[pair_index] if pair_index < len(tgt_pages) else None
        if _belongs_to_pair_slice(result, sp=sp, tp=tp):
            return pair_index
    return -1


def build_visual_diff_artifacts(
    response: CompareResponse,
    compare_artifacts: CompareArtifacts,
) -> VisualDiffArtifacts:
    """Build manifest, reviewer PDF, flattened PNG image, and debug HTML report."""
    manifest = build_visual_diff_manifest(response, compare_artifacts)
    overlay_pdf = render_visual_diff_overlay_pdf_pages(manifest, compare_artifacts)
    overlay_bgr = render_visual_diff_overlay_pages(manifest, compare_artifacts)
    html_report = build_visual_diff_html_pages(manifest, compare_artifacts)
    return VisualDiffArtifacts(
        manifest=manifest,
        overlay_pdf=overlay_pdf,
        overlay_bgr=overlay_bgr,
        html=html_report,
    )


def build_visual_diff_manifest(
    response: CompareResponse,
    compare_artifacts: CompareArtifacts,
) -> VisualDiffManifest:
    """Convert non-exact `MatchResult` rows into a stable visual manifest."""
    src_pages = compare_artifacts.source_pages
    tgt_pages = compare_artifacts.target_pages
    if not src_pages or not tgt_pages:
        raise ValueError("CompareArtifacts requires at least one source and one target page")

    n_pairs = max(len(src_pages), len(tgt_pages))
    slice_ann: list[list[VisualAnnotation]] = [[] for _ in range(n_pairs)]
    flat: list[VisualAnnotation] = []
    idx_global = 0

    for result in _iter_visual_results(compare_artifacts.results):
        pair_index = _pair_index_for_result(result, src_pages=src_pages, tgt_pages=tgt_pages)
        if pair_index < 0:
            continue
        sp = src_pages[pair_index] if pair_index < len(src_pages) else None
        tp = tgt_pages[pair_index] if pair_index < len(tgt_pages) else None
        sw = int(sp.processed_bgr.shape[1]) if sp is not None else 0
        sh = int(sp.processed_bgr.shape[0]) if sp is not None else 0
        tw = int(tp.processed_bgr.shape[1]) if tp is not None else 0
        th = int(tp.processed_bgr.shape[0]) if tp is not None else 0
        idx_global += 1
        ann = _build_annotation(
            result,
            index=idx_global,
            source_width=sw if sw > 0 else 1,
            source_height=sh if sh > 0 else 1,
            target_width=tw if tw > 0 else 1,
            target_height=th if th > 0 else 1,
        )
        flat.append(ann)
        slice_ann[pair_index].append(ann)

    pages: list[VisualDiffPageSlice] = []
    for pair_index in range(n_pairs):
        sp = src_pages[pair_index] if pair_index < len(src_pages) else None
        tp = tgt_pages[pair_index] if pair_index < len(tgt_pages) else None
        sw = int(sp.processed_bgr.shape[1]) if sp is not None else 0
        sh = int(sp.processed_bgr.shape[0]) if sp is not None else 0
        tw = int(tp.processed_bgr.shape[1]) if tp is not None else 0
        th = int(tp.processed_bgr.shape[0]) if tp is not None else 0
        pages.append(
            VisualDiffPageSlice(
                pair_index=pair_index,
                source_page_number=sp.page_number if sp is not None else 0,
                target_page_number=tp.page_number if tp is not None else 0,
                source_image_width=sw,
                source_image_height=sh,
                target_image_width=tw,
                target_image_height=th,
                annotations=slice_ann[pair_index],
            )
        )

    p0 = pages[0]
    return VisualDiffManifest(
        comparison_id=str(response.comparison_id),
        source_path=str(response.extras.get("source_path", "")),
        target_path=str(response.extras.get("target_path", "")),
        ocr_provider=str(response.extras.get("ocr_provider", "")),
        source_page_number=p0.source_page_number or 1,
        target_page_number=p0.target_page_number or 1,
        source_image_width=max(p0.source_image_width, 1),
        source_image_height=max(p0.source_image_height, 1),
        target_image_width=max(p0.target_image_width, 1),
        target_image_height=max(p0.target_image_height, 1),
        summary=response.report.summary,
        annotations=flat,
        pages=pages,
    )


def plan_badge_placements_for_canvas(
    width: int,
    height: int,
    annotations: list[VisualAnnotation],
) -> list[BadgePlacement]:
    """Badge layout for one canvas of size ``width`` x ``height``."""
    gap = _badge_gap(width, height)
    placements: list[BadgePlacement] = []
    occupied: list[tuple[int, int, int, int]] = []
    for annotation in annotations:
        box = _normalized_box_to_pixels(
            annotation.overlay_bbox_normalized,
            width=width,
            height=height,
        )
        badge_w, badge_h = _badge_dimensions(str(annotation.index), width, height)
        placement = _place_badge_for_box(
            annotation_id=annotation.annotation_id,
            index=annotation.index,
            box=box,
            badge_w=badge_w,
            badge_h=badge_h,
            gap=gap,
            page_width=width,
            page_height=height,
            occupied=occupied,
        )
        occupied.append((placement.x1, placement.y1, placement.x2, placement.y2))
        placements.append(placement)
    return placements


def plan_badge_placements(manifest: VisualDiffManifest) -> list[BadgePlacement]:
    """Place numbered badges; multi-page manifests concatenate per-slice placements in manifest order."""
    if not manifest.pages:
        return plan_badge_placements_for_canvas(
            manifest.source_image_width,
            manifest.source_image_height,
            manifest.annotations,
        )
    out: list[BadgePlacement] = []
    for sl in manifest.pages:
        cw = sl.source_image_width or sl.target_image_width
        ch = sl.source_image_height or sl.target_image_height
        if cw <= 0 or ch <= 0:
            continue
        out.extend(plan_badge_placements_for_canvas(cw, ch, sl.annotations))
    return out


def render_visual_diff_overlay(
    manifest: VisualDiffManifest,
    *,
    source_image: BGRImage,
    target_image: BGRImage,
) -> BGRImage:
    """Flatten the source image plus all visible highlights into one preview raster."""
    canvas = source_image.copy()
    height, width = canvas.shape[:2]
    thickness = _line_thickness(width, height)
    font_scale = _font_scale(width, height)
    placements = {
        p.annotation_id: p
        for p in plan_badge_placements_for_canvas(width, height, manifest.annotations)
    }
    for annotation in manifest.annotations:
        style = _style_for(annotation.match_type)
        x1, y1, x2, y2 = _normalized_box_to_pixels(
            annotation.overlay_bbox_normalized,
            width=width,
            height=height,
        )
        if annotation.match_type in _TARGET_CROP_MATCH_TYPES and annotation.target_bbox is not None:
            crop = _crop_image(target_image, annotation.target_bbox)
            if crop is not None:
                _blend_crop(canvas, crop, x1=x1, y1=y1, x2=x2, y2=y2, alpha=style.crop_alpha)
        if style.fill_alpha > 0.0:
            _fill_box(canvas, x1=x1, y1=y1, x2=x2, y2=y2, bgr=style.bgr, alpha=style.fill_alpha)
        _draw_box(canvas, x1=x1, y1=y1, x2=x2, y2=y2, bgr=style.bgr, thickness=thickness)
        _draw_badge(
            canvas,
            text=str(annotation.index),
            placement=placements[annotation.annotation_id],
            bgr=style.bgr,
            font_scale=font_scale,
            thickness=thickness,
        )
    return canvas


def _render_single_page_overlay(
    annotations: list[VisualAnnotation],
    *,
    source_image: BGRImage | None,
    target_image: BGRImage | None,
) -> BGRImage:
    if source_image is not None:
        canvas = source_image.copy()
    elif target_image is not None:
        canvas = target_image.copy()
    else:
        raise ValueError("source_image or target_image required")
    height, width = canvas.shape[:2]
    thickness = _line_thickness(width, height)
    font_scale = _font_scale(width, height)
    placements = {
        p.annotation_id: p for p in plan_badge_placements_for_canvas(width, height, annotations)
    }
    for annotation in annotations:
        style = _style_for(annotation.match_type)
        x1, y1, x2, y2 = _normalized_box_to_pixels(
            annotation.overlay_bbox_normalized,
            width=width,
            height=height,
        )
        if (
            annotation.match_type in _TARGET_CROP_MATCH_TYPES
            and annotation.target_bbox is not None
            and target_image is not None
        ):
            crop = _crop_image(target_image, annotation.target_bbox)
            if crop is not None:
                _blend_crop(canvas, crop, x1=x1, y1=y1, x2=x2, y2=y2, alpha=style.crop_alpha)
        if style.fill_alpha > 0.0:
            _fill_box(canvas, x1=x1, y1=y1, x2=x2, y2=y2, bgr=style.bgr, alpha=style.fill_alpha)
        _draw_box(canvas, x1=x1, y1=y1, x2=x2, y2=y2, bgr=style.bgr, thickness=thickness)
        _draw_badge(
            canvas,
            text=str(annotation.index),
            placement=placements[annotation.annotation_id],
            bgr=style.bgr,
            font_scale=font_scale,
            thickness=thickness,
        )
    return canvas


def render_visual_diff_overlay_pages(
    manifest: VisualDiffManifest,
    compare_artifacts: CompareArtifacts,
) -> BGRImage:
    """Stack one overlay raster per aligned page pair (vertical stack for multi-page)."""
    bands: list[np.ndarray] = []
    for sl in manifest.pages:
        sp = (
            compare_artifacts.source_pages[sl.pair_index]
            if sl.pair_index < len(compare_artifacts.source_pages)
            else None
        )
        tp = (
            compare_artifacts.target_pages[sl.pair_index]
            if sl.pair_index < len(compare_artifacts.target_pages)
            else None
        )
        src_img = sp.processed_bgr if sp is not None else None
        tgt_img = tp.processed_bgr if tp is not None else None
        if src_img is None and tgt_img is None:
            continue
        bands.append(_render_single_page_overlay(sl.annotations, source_image=src_img, target_image=tgt_img))
    if not bands:
        return compare_artifacts.source_page.processed_bgr.copy()
    if len(bands) == 1:
        return bands[0]
    return np.vstack(bands)


def render_visual_diff_overlay_pdf(
    manifest: VisualDiffManifest,
    *,
    source_image: BGRImage,
) -> bytes:
    """Build the reviewer-facing annotated PDF from the processed source page image."""
    doc = fitz.open()
    try:
        page = doc.new_page(
            width=float(manifest.source_image_width),
            height=float(manifest.source_image_height),
        )
        page.insert_image(
            fitz.Rect(0.0, 0.0, float(manifest.source_image_width), float(manifest.source_image_height)),
            stream=image_to_png_bytes(source_image),
            overlay=False,
        )
        placements = {
            p.annotation_id: p
            for p in plan_badge_placements_for_canvas(
                manifest.source_image_width,
                manifest.source_image_height,
                manifest.annotations,
            )
        }
        line_width = _pdf_line_width(manifest.source_image_width, manifest.source_image_height)
        badge_font_size = _badge_font_size(manifest.source_image_width, manifest.source_image_height)
        for annotation in manifest.annotations:
            style = _style_for(annotation.match_type)
            x1, y1, x2, y2 = _normalized_box_to_pixels(
                annotation.overlay_bbox_normalized,
                width=manifest.source_image_width,
                height=manifest.source_image_height,
            )
            page.draw_rect(
                fitz.Rect(float(x1), float(y1), float(x2), float(y2)),
                color=style.pdf_rgb,
                fill=style.pdf_rgb,
                width=line_width,
                fill_opacity=style.pdf_fill_alpha,
                stroke_opacity=1.0,
                overlay=True,
            )
            placement = placements[annotation.annotation_id]
            badge_rect = fitz.Rect(
                float(placement.x1),
                float(placement.y1),
                float(placement.x2),
                float(placement.y2),
            )
            page.draw_rect(
                badge_rect,
                color=style.pdf_rgb,
                fill=style.pdf_rgb,
                width=max(0.8, line_width * 0.75),
                fill_opacity=1.0,
                stroke_opacity=1.0,
                overlay=True,
            )
            _insert_badge_text(
                page,
                badge_rect=badge_rect,
                text=str(annotation.index),
                font_size=badge_font_size,
            )
        return doc.write(deflate=True)
    finally:
        doc.close()


def render_visual_diff_overlay_pdf_pages(
    manifest: VisualDiffManifest,
    compare_artifacts: CompareArtifacts,
) -> bytes:
    """One PDF page per aligned pair, each with annotations on the source (or target-only) raster."""
    doc = fitz.open()
    try:
        for sl in manifest.pages:
            sp = (
                compare_artifacts.source_pages[sl.pair_index]
                if sl.pair_index < len(compare_artifacts.source_pages)
                else None
            )
            tp = (
                compare_artifacts.target_pages[sl.pair_index]
                if sl.pair_index < len(compare_artifacts.target_pages)
                else None
            )
            src_img = sp.processed_bgr if sp is not None else None
            tgt_img = tp.processed_bgr if tp is not None else None
            base = src_img if src_img is not None else tgt_img
            if base is None:
                continue
            bw, bh = int(base.shape[1]), int(base.shape[0])
            page = doc.new_page(width=float(bw), height=float(bh))
            page.insert_image(
                fitz.Rect(0.0, 0.0, float(bw), float(bh)),
                stream=image_to_png_bytes(base),
                overlay=False,
            )
            placements = {p.annotation_id: p for p in plan_badge_placements_for_canvas(bw, bh, sl.annotations)}
            line_width = _pdf_line_width(bw, bh)
            badge_font_size = _badge_font_size(bw, bh)
            for annotation in sl.annotations:
                style = _style_for(annotation.match_type)
                x1, y1, x2, y2 = _normalized_box_to_pixels(
                    annotation.overlay_bbox_normalized,
                    width=bw,
                    height=bh,
                )
                page.draw_rect(
                    fitz.Rect(float(x1), float(y1), float(x2), float(y2)),
                    color=style.pdf_rgb,
                    fill=style.pdf_rgb,
                    width=line_width,
                    fill_opacity=style.pdf_fill_alpha,
                    stroke_opacity=1.0,
                    overlay=True,
                )
                placement = placements[annotation.annotation_id]
                badge_rect = fitz.Rect(
                    float(placement.x1),
                    float(placement.y1),
                    float(placement.x2),
                    float(placement.y2),
                )
                page.draw_rect(
                    badge_rect,
                    color=style.pdf_rgb,
                    fill=style.pdf_rgb,
                    width=max(0.8, line_width * 0.75),
                    fill_opacity=1.0,
                    stroke_opacity=1.0,
                    overlay=True,
                )
                _insert_badge_text(
                    page,
                    badge_rect=badge_rect,
                    text=str(annotation.index),
                    font_size=badge_font_size,
                )
        return doc.write(deflate=True)
    finally:
        doc.close()


def build_visual_diff_html(
    manifest: VisualDiffManifest,
    *,
    source_image: BGRImage,
    target_image: BGRImage,
) -> str:
    """Build a self-contained interactive HTML report with canvas-based toggles."""
    source_name = Path(manifest.source_path).name or "(source drawing)"
    target_name = Path(manifest.target_path).name or "(target drawing)"
    payload = _html_payload(manifest, source_image=source_image, target_image=target_image)
    rows_html = _table_rows_markup(manifest)
    non_exact = (
        manifest.summary.changed
        + manifest.summary.missing
        + manifest.summary.extra_target
        + manifest.summary.uncertain
    )
    rows_or_empty = (
        _empty_table_markup()
        if not manifest.annotations
        else (
            "<table><thead><tr><th>#</th><th>Type</th><th>Source text</th>"
            "<th>Target text</th><th>Confidence</th><th>Reason</th></tr></thead><tbody>"
            + rows_html
            + "</tbody></table>"
        )
    )
    return (
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Visual Overlay Diff Report</title>
  <style>
    :root {
      --paper: #faf7f1;
      --ink: #16202b;
      --muted: #5d6970;
      --line: rgba(22, 32, 43, 0.12);
      --accent: #0b5c80;
      --panel: rgba(255,255,255,0.82);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(11, 92, 128, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.14), transparent 28%),
        linear-gradient(180deg, #edf3f6 0%, var(--paper) 45%, #f2eee5 100%);
      font-family: "Bahnschrift", "Aptos", "Trebuchet MS", sans-serif;
    }
    .page { width: min(1380px, calc(100vw - 24px)); margin: 18px auto 32px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 36px rgba(22, 32, 43, 0.12);
      padding: 20px;
      margin-bottom: 16px;
    }
    h1, h2 {
      margin: 0 0 10px;
      font-family: "Rockwell", "Aptos Display", Georgia, serif;
    }
    .lede { color: var(--muted); max-width: 860px; line-height: 1.55; margin: 0; }
    .meta, .stats {
      display: grid;
      gap: 10px;
      margin-top: 16px;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }
    .pill, .stat {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.74);
    }
    .k {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .v { font-weight: 700; word-break: break-word; }
    .mono { font-family: "Consolas", "Cascadia Mono", monospace; font-size: 13px; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; }
    .legend label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.9);
      cursor: pointer;
    }
    .legend input { accent-color: var(--accent); }
    .swatch {
      width: 11px;
      height: 11px;
      border-radius: 999px;
      display: inline-block;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.15);
    }
    .viewer {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background:
        linear-gradient(180deg, rgba(11, 92, 128, 0.06), rgba(255,255,255,0.92)),
        repeating-linear-gradient(45deg, rgba(22, 32, 43, 0.02), rgba(22, 32, 43, 0.02) 8px, transparent 8px, transparent 16px);
    }
    canvas {
      width: 100%;
      height: auto;
      max-height: 78vh;
      display: block;
      border-radius: 12px;
      background: white;
    }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; color: var(--muted); }
    tbody tr:hover { background: rgba(11, 92, 128, 0.04); }
    .note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin-top: 8px;
    }
    .empty {
      padding: 14px;
      border-radius: 14px;
      border: 1px dashed rgba(11, 92, 128, 0.28);
      background: rgba(11, 92, 128, 0.06);
      color: var(--muted);
    }
    @media (max-width: 720px) {
      .page { width: min(100vw - 12px, 1380px); margin-top: 8px; }
      .card { padding: 16px; }
      .toolbar { align-items: flex-start; }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="card">
      <h1>Visual Overlay Diff Report</h1>
      <p class="lede">The source drawing is the base canvas. Non-exact findings are projected into source coordinates and highlighted with mapped target overlays or translucent fills, so reviewers can scan changes without leaving the drawing.</p>
      <div class="meta">
        <div class="pill"><div class="k">Source drawing</div><div class="v mono">__SOURCE_NAME__</div></div>
        <div class="pill"><div class="k">Target drawing</div><div class="v mono">__TARGET_NAME__</div></div>
        <div class="pill"><div class="k">Comparison ID</div><div class="v mono">__COMPARISON_ID__</div></div>
        <div class="pill"><div class="k">OCR provider</div><div class="v">__OCR_PROVIDER__</div></div>
      </div>
    </section>

    <section class="card">
      <h2>Summary</h2>
      <div class="stats">
        <div class="stat"><div class="k">Changed</div><div class="v">__CHANGED__</div></div>
        <div class="stat"><div class="k">Missing</div><div class="v">__MISSING__</div></div>
        <div class="stat"><div class="k">Only on target</div><div class="v">__EXTRA__</div></div>
        <div class="stat"><div class="k">Unclear</div><div class="v">__UNCERTAIN__</div></div>
        <div class="stat"><div class="k">Visible annotations</div><div class="v" id="visibleCount">__VISIBLE_COUNT__</div></div>
      </div>
    </section>

    <section class="card">
      <div class="toolbar">
        <div>
          <h2>Canvas Review</h2>
          <div class="note">Toggle finding types to simplify the view. Box numbers match the findings table below.</div>
        </div>
        <div class="legend">__LEGEND__</div>
      </div>
      <div class="viewer">
        <div class="note">Mapped highlights: <strong>__NON_EXACT__</strong> non-exact finding(s). Canvas size: <span class="mono">__SOURCE_WIDTH__ x __SOURCE_HEIGHT__</span></div>
        <canvas id="diffCanvas" width="__SOURCE_WIDTH__" height="__SOURCE_HEIGHT__"></canvas>
      </div>
    </section>

    <section class="card">
      <h2>Findings</h2>
      __ROWS_OR_EMPTY__
    </section>
  </div>

  <script id="visual-diff-data" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const payload = JSON.parse(document.getElementById("visual-diff-data").textContent);
    const canvas = document.getElementById("diffCanvas");
    const ctx = canvas.getContext("2d");
    const visibleCount = document.getElementById("visibleCount");
    const inputs = Array.from(document.querySelectorAll("[data-toggle-type]"));
    const rows = Array.from(document.querySelectorAll("tbody tr[data-match-type]"));

    function loadImage(dataUrl) {
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = dataUrl;
      });
    }

    function activeTypes() {
      return new Set(inputs.filter((input) => input.checked).map((input) => input.dataset.toggleType));
    }

    function drawBadge(badgeBox, text, color) {
      const x = Math.round(badgeBox.x1 * canvas.width);
      const y = Math.round(badgeBox.y1 * canvas.height);
      const w = Math.max(1, Math.round((badgeBox.x2 - badgeBox.x1) * canvas.width));
      const h = Math.max(1, Math.round((badgeBox.y2 - badgeBox.y1) * canvas.height));
      const fontPx = Math.max(10, Math.min(h - 4, Math.round(h * 0.66)));
      ctx.save();
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.font = `700 ${fontPx}px Bahnschrift, Trebuchet MS, sans-serif`;
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillText(text, x + w / 2, y + h / 2 + 1);
      ctx.restore();
    }

    function updateRows(types) {
      let shown = 0;
      rows.forEach((row) => {
        const visible = types.has(row.dataset.matchType);
        row.hidden = !visible;
        if (visible) shown += 1;
      });
      visibleCount.textContent = String(shown);
    }

    async function init() {
      const baseImage = await loadImage(payload.base_image_data_url);
      const cropImages = {};
      await Promise.all(payload.annotations.map(async (ann) => {
        if (ann.target_crop_data_url) {
          cropImages[ann.annotation_id] = await loadImage(ann.target_crop_data_url);
        }
      }));

      function redraw() {
        const types = activeTypes();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
        payload.annotations.forEach((ann) => {
          if (!types.has(ann.match_type)) return;
          const box = ann.overlay_bbox_normalized;
          const x = Math.round(box.x1 * canvas.width);
          const y = Math.round(box.y1 * canvas.height);
          const w = Math.max(1, Math.round((box.x2 - box.x1) * canvas.width));
          const h = Math.max(1, Math.round((box.y2 - box.y1) * canvas.height));
          if (ann.target_crop_data_url && cropImages[ann.annotation_id]) {
            ctx.save();
            ctx.globalAlpha = ann.crop_alpha;
            ctx.drawImage(cropImages[ann.annotation_id], x, y, w, h);
            ctx.restore();
          }
          if (ann.fill_alpha > 0) {
            ctx.save();
            ctx.globalAlpha = ann.fill_alpha;
            ctx.fillStyle = ann.color_hex;
            ctx.fillRect(x, y, w, h);
            ctx.restore();
          }
          ctx.save();
          ctx.strokeStyle = ann.color_hex;
          ctx.lineWidth = Math.max(2, Math.round(Math.max(canvas.width, canvas.height) / 900));
          ctx.strokeRect(x, y, w, h);
          ctx.restore();
          if (ann.badge_bbox_normalized) {
            drawBadge(ann.badge_bbox_normalized, String(ann.index), ann.color_hex);
          }
        });
        updateRows(types);
      }

      inputs.forEach((input) => input.addEventListener("change", redraw));
      redraw();
    }

    init().catch((error) => console.error("visual diff init failed", error));
  </script>
</body>
</html>
"""
        .replace("__SOURCE_NAME__", html.escape(source_name))
        .replace("__TARGET_NAME__", html.escape(target_name))
        .replace("__COMPARISON_ID__", html.escape(manifest.comparison_id))
        .replace("__OCR_PROVIDER__", html.escape(manifest.ocr_provider or "(default)"))
        .replace("__CHANGED__", str(manifest.summary.changed))
        .replace("__MISSING__", str(manifest.summary.missing))
        .replace("__EXTRA__", str(manifest.summary.extra_target))
        .replace("__UNCERTAIN__", str(manifest.summary.uncertain))
        .replace("__VISIBLE_COUNT__", str(manifest.annotation_count))
        .replace("__NON_EXACT__", str(non_exact))
        .replace("__SOURCE_WIDTH__", str(manifest.source_image_width))
        .replace("__SOURCE_HEIGHT__", str(manifest.source_image_height))
        .replace("__LEGEND__", _legend_markup(manifest))
        .replace("__ROWS_OR_EMPTY__", rows_or_empty)
        .replace("__PAYLOAD_JSON__", _safe_script_json(payload))
    )


def _html_annotation_dict(
    annotation: VisualAnnotation,
    placement: BadgePlacement,
    *,
    canvas_width: int,
    canvas_height: int,
    target_image: BGRImage | None,
) -> dict[str, object]:
    style = _style_for(annotation.match_type)
    crop_data_url = None
    if (
        annotation.target_bbox is not None
        and annotation.match_type in _TARGET_CROP_MATCH_TYPES
        and target_image is not None
    ):
        crop = _crop_image(target_image, annotation.target_bbox)
        if crop is not None:
            crop_data_url = image_to_data_url(crop)
    return {
        "annotation_id": annotation.annotation_id,
        "index": annotation.index,
        "match_type": annotation.match_type.value,
        "match_label": style.label,
        "color_hex": style.color_hex,
        "fill_alpha": style.fill_alpha,
        "crop_alpha": style.crop_alpha,
        "source_text": annotation.source_text,
        "target_text": annotation.target_text,
        "confidence": round(annotation.confidence, 4),
        "reason": annotation.reason or "",
        "overlay_bbox_normalized": annotation.overlay_bbox_normalized.model_dump(mode="json"),
        "badge_bbox_normalized": placement.normalized_bbox(
            page_width=canvas_width,
            page_height=canvas_height,
        ).model_dump(mode="json"),
        "badge_placement": placement.placement,
        "target_crop_data_url": crop_data_url,
    }


def _html_payload_page_slice(
    manifest: VisualDiffManifest,
    sl: VisualDiffPageSlice,
    *,
    source_image: BGRImage | None,
    target_image: BGRImage | None,
) -> dict[str, object]:
    cw = sl.source_image_width or sl.target_image_width
    ch = sl.source_image_height or sl.target_image_height
    placements = {p.annotation_id: p for p in plan_badge_placements_for_canvas(cw, ch, sl.annotations)}
    base = source_image if source_image is not None else target_image
    if base is None:
        raise ValueError("page slice requires a source or target raster")
    ann_dicts: list[dict[str, object]] = []
    for annotation in sl.annotations:
        ann_dicts.append(
            _html_annotation_dict(
                annotation,
                placements[annotation.annotation_id],
                canvas_width=cw,
                canvas_height=ch,
                target_image=target_image,
            )
        )
    return {
        "pair_index": sl.pair_index,
        "source_page_number": sl.source_page_number,
        "target_page_number": sl.target_page_number,
        "canvas_width": cw,
        "canvas_height": ch,
        "base_image_data_url": image_to_data_url(base),
        "annotations": ann_dicts,
    }


def build_visual_diff_html_pages(
    manifest: VisualDiffManifest,
    compare_artifacts: CompareArtifacts,
) -> str:
    """HTML report: single-page uses legacy template; multi-page adds one canvas per aligned pair."""
    if len(manifest.pages) <= 1:
        sp = compare_artifacts.source_pages[0]
        tp = compare_artifacts.target_pages[0]
        return build_visual_diff_html(
            manifest,
            source_image=sp.processed_bgr,
            target_image=tp.processed_bgr,
        )
    return _build_visual_diff_html_multipage(manifest, compare_artifacts)


def _build_visual_diff_html_multipage(
    manifest: VisualDiffManifest,
    compare_artifacts: CompareArtifacts,
) -> str:
    source_name = Path(manifest.source_path).name or "(source drawing)"
    target_name = Path(manifest.target_path).name or "(target drawing)"
    page_payloads: list[dict[str, object]] = []
    canvas_blocks: list[str] = []
    for sl in manifest.pages:
        sp = (
            compare_artifacts.source_pages[sl.pair_index]
            if sl.pair_index < len(compare_artifacts.source_pages)
            else None
        )
        tp = (
            compare_artifacts.target_pages[sl.pair_index]
            if sl.pair_index < len(compare_artifacts.target_pages)
            else None
        )
        src_img = sp.processed_bgr if sp is not None else None
        tgt_img = tp.processed_bgr if tp is not None else None
        cw = sl.source_image_width or sl.target_image_width
        ch = sl.source_image_height or sl.target_image_height
        if cw <= 0 or ch <= 0:
            continue
        page_payloads.append(
            _html_payload_page_slice(manifest, sl, source_image=src_img, target_image=tgt_img)
        )
        idx = sl.pair_index
        label = f"Pair {idx + 1}: source p.{sl.source_page_number or '—'} / target p.{sl.target_page_number or '—'}"
        canvas_blocks.append(
            f'<section class="card"><h2>{html.escape(label)}</h2>'
            f'<div class="note">Canvas {cw} × {ch} px</div>'
            f'<div class="viewer"><canvas id="diffCanvas-{idx}" width="{cw}" height="{ch}"></canvas></div></section>'
        )

    payload: dict[str, object] = {
        "schema_version": "2.0",
        "manifest": manifest.model_dump(mode="json"),
        "pages": page_payloads,
    }
    rows_html = _table_rows_markup(manifest)
    non_exact = (
        manifest.summary.changed
        + manifest.summary.missing
        + manifest.summary.extra_target
        + manifest.summary.uncertain
    )
    rows_or_empty = (
        _empty_table_markup()
        if not manifest.annotations
        else (
            "<table><thead><tr><th>#</th><th>Type</th><th>Source text</th>"
            "<th>Target text</th><th>Confidence</th><th>Reason</th></tr></thead><tbody>"
            + rows_html
            + "</tbody></table>"
        )
    )
    blocks_html = "\n".join(canvas_blocks)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Visual Overlay Diff Report (multi-page)</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f0f4f8; color: #16202b; }}
    .page {{ width: min(1380px, calc(100vw - 24px)); margin: 18px auto 32px; }}
    .card {{ background: #fff; border-radius: 12px; padding: 16px; margin-bottom: 16px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
    .viewer {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; background: #fafafa; }}
    canvas {{ max-width: 100%; height: auto; display: block; }}
    .note {{ color: #5d6970; font-size: 13px; margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #eee; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="card">
      <h1>Visual Overlay Diff Report</h1>
      <p>Multi-page compare: one canvas per aligned page index (source N vs target N).</p>
      <div><strong>Source</strong> <span class="mono">{html.escape(source_name)}</span></div>
      <div><strong>Target</strong> <span class="mono">{html.escape(target_name)}</span></div>
      <div><strong>Comparison ID</strong> <span class="mono">{html.escape(manifest.comparison_id)}</span></div>
    </section>
    <section class="card">
      <h2>Summary</h2>
      <p>Changed {manifest.summary.changed}, Missing {manifest.summary.missing},
      Extra {manifest.summary.extra_target}, Uncertain {manifest.summary.uncertain}</p>
    </section>
    <section class="card">
      <h2>Findings</h2>
      {rows_or_empty}
    </section>
    {blocks_html}
  </div>
  <script id="visual-diff-data" type="application/json">{_safe_script_json(payload)}</script>
  <script>
    const payload = JSON.parse(document.getElementById("visual-diff-data").textContent);
    function loadImage(dataUrl) {{
      return new Promise((resolve, reject) => {{
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = dataUrl;
      }});
    }}
    function drawBadge(ctx, canvas, badgeBox, text, color) {{
      const x = Math.round(badgeBox.x1 * canvas.width);
      const y = Math.round(badgeBox.y1 * canvas.height);
      const w = Math.max(1, Math.round((badgeBox.x2 - badgeBox.x1) * canvas.width));
      const h = Math.max(1, Math.round((badgeBox.y2 - badgeBox.y1) * canvas.height));
      const fontPx = Math.max(10, Math.min(h - 4, Math.round(h * 0.66)));
      ctx.save();
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.font = `700 ${{fontPx}}px sans-serif`;
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillText(text, x + w / 2, y + h / 2 + 1);
      ctx.restore();
    }}
    async function initPage(page) {{
      const canvas = document.getElementById("diffCanvas-" + page.pair_index);
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const baseImage = await loadImage(page.base_image_data_url);
      const cropImages = {{}};
      await Promise.all(page.annotations.map(async (ann) => {{
        if (ann.target_crop_data_url) {{
          cropImages[ann.annotation_id] = await loadImage(ann.target_crop_data_url);
        }}
      }}));
      function redraw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
        page.annotations.forEach((ann) => {{
          const box = ann.overlay_bbox_normalized;
          const x = Math.round(box.x1 * canvas.width);
          const y = Math.round(box.y1 * canvas.height);
          const w = Math.max(1, Math.round((box.x2 - box.x1) * canvas.width));
          const h = Math.max(1, Math.round((box.y2 - box.y1) * canvas.height));
          if (ann.target_crop_data_url && cropImages[ann.annotation_id]) {{
            ctx.save();
            ctx.globalAlpha = ann.crop_alpha;
            ctx.drawImage(cropImages[ann.annotation_id], x, y, w, h);
            ctx.restore();
          }}
          if (ann.fill_alpha > 0) {{
            ctx.save();
            ctx.globalAlpha = ann.fill_alpha;
            ctx.fillStyle = ann.color_hex;
            ctx.fillRect(x, y, w, h);
            ctx.restore();
          }}
          ctx.save();
          ctx.strokeStyle = ann.color_hex;
          ctx.lineWidth = Math.max(2, Math.round(Math.max(canvas.width, canvas.height) / 900));
          ctx.strokeRect(x, y, w, h);
          ctx.restore();
          if (ann.badge_bbox_normalized) {{
            drawBadge(ctx, canvas, ann.badge_bbox_normalized, String(ann.index), ann.color_hex);
          }}
        }});
      }}
      redraw();
    }}
    async function init() {{
      for (const p of payload.pages) {{
        await initPage(p);
      }}
    }}
    init().catch((e) => console.error(e));
  </script>
</body>
</html>"""


def image_to_png_bytes(image: BGRImage) -> bytes:
    """Encode a BGR image to PNG bytes."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return bytes(encoded.tobytes())


def image_to_data_url(image: BGRImage) -> str:
    """Encode a BGR image as an inline PNG data URL."""
    return "data:image/png;base64," + base64.b64encode(image_to_png_bytes(image)).decode("ascii")


def _build_annotation(
    result: MatchResult,
    *,
    index: int,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> VisualAnnotation:
    src = result.source_field
    tgt = result.target_field
    src_norm = (
        NormalizedBoundingBox.from_bbox(src.bbox, width=source_width, height=source_height)
        if src is not None
        else None
    )
    tgt_norm = (
        NormalizedBoundingBox.from_bbox(tgt.bbox, width=target_width, height=target_height)
        if tgt is not None
        else None
    )
    return VisualAnnotation(
        annotation_id=f"ann-{index:03d}",
        index=index,
        match_type=result.match_type,
        source_field_id=src.field_id if src is not None else None,
        target_field_id=tgt.field_id if tgt is not None else None,
        source_text=field_caption(src) if src is not None else "",
        target_text=field_caption(tgt) if tgt is not None else "",
        confidence=result.confidence,
        reason=result.reason,
        source_bbox=src.bbox if src is not None else None,
        target_bbox=tgt.bbox if tgt is not None else None,
        source_bbox_normalized=src_norm,
        target_bbox_normalized=tgt_norm,
        overlay_bbox_normalized=_overlay_bbox_for_result(
            result=result,
            source_box=src_norm,
            target_box=tgt_norm,
        ),
    )


def _overlay_bbox_for_result(
    *,
    result: MatchResult,
    source_box: NormalizedBoundingBox | None,
    target_box: NormalizedBoundingBox | None,
) -> NormalizedBoundingBox:
    if result.match_type == MatchType.MISSING_IN_TARGET and source_box is not None:
        return source_box
    if result.match_type in _TARGET_CROP_MATCH_TYPES and target_box is not None:
        return target_box
    if source_box is not None:
        return source_box
    if target_box is not None:
        return target_box
    raise ValueError(f"Could not determine overlay box for {result.match_type!r}")


def _iter_visual_results(results: list[MatchResult]) -> list[MatchResult]:
    return [r for r in results if r.match_type in _VISIBLE_MATCH_TYPES]


def _fill_box(
    image: BGRImage,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    tint = np.full_like(roi, bgr, dtype=np.uint8)
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _blend_crop(
    image: BGRImage,
    crop: BGRImage,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    alpha: float,
) -> None:
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    interpolation = cv2.INTER_AREA if crop.shape[0] >= roi.shape[0] else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (roi.shape[1], roi.shape[0]), interpolation=interpolation)
    cv2.addWeighted(resized, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _draw_box(
    image: BGRImage,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.rectangle(
        image,
        (x1, y1),
        (max(x1, x2 - 1), max(y1, y2 - 1)),
        bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_badge(
    image: BGRImage,
    *,
    text: str,
    placement: BadgePlacement,
    bgr: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, thickness)
    max_text_w = max(1, placement.width - 6)
    max_text_h = max(1, placement.height - 4)
    local_scale = font_scale
    (tw, th), baseline = cv2.getTextSize(text, font, local_scale, text_thickness)
    while local_scale > 0.25 and (tw > max_text_w or th + baseline > max_text_h):
        local_scale = max(0.25, local_scale - 0.05)
        (tw, th), baseline = cv2.getTextSize(text, font, local_scale, text_thickness)
    cv2.rectangle(
        image,
        (placement.x1, placement.y1),
        (placement.x2, placement.y2),
        bgr,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    text_x = placement.x1 + max(1, (placement.width - tw) // 2)
    text_y = placement.y1 + max(th, (placement.height + th - baseline) // 2)
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        local_scale,
        (255, 255, 255),
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )


def _insert_badge_text(
    page: fitz.Page,
    *,
    badge_rect: fitz.Rect,
    text: str,
    font_size: float,
) -> None:
    inset_x = max(1.5, badge_rect.width * 0.18)
    inset_y = max(1.0, badge_rect.height * 0.08)
    text_rect = fitz.Rect(
        badge_rect.x0 + inset_x,
        badge_rect.y0 + inset_y,
        badge_rect.x1 - inset_x,
        badge_rect.y1 - inset_y,
    )
    local_size = font_size
    while local_size >= 6.0:
        written = page.insert_textbox(
            text_rect,
            text,
            fontname="helv",
            fontsize=local_size,
            color=(1.0, 1.0, 1.0),
            align=1,
            overlay=True,
        )
        if written >= 0:
            return
        local_size -= 1.0
    page.insert_textbox(
        badge_rect,
        text,
        fontname="helv",
        fontsize=6.0,
        color=(1.0, 1.0, 1.0),
        align=1,
        overlay=True,
    )


def _crop_image(image: BGRImage, bbox: BoundingBox) -> BGRImage | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _bbox_to_pixel_bounds(bbox, width=w, height=h)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def _bbox_to_pixel_bounds(
    bbox: BoundingBox,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(np.floor(bbox.x1))))
    y1 = max(0, min(height - 1, int(np.floor(bbox.y1))))
    x2 = max(x1 + 1, min(width, int(np.ceil(bbox.x2))))
    y2 = max(y1 + 1, min(height, int(np.ceil(bbox.y2))))
    return x1, y1, x2, y2


def _normalized_box_to_pixels(
    box: NormalizedBoundingBox,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(np.floor(box.x1 * width))))
    y1 = max(0, min(height - 1, int(np.floor(box.y1 * height))))
    x2 = max(x1 + 1, min(width, int(np.ceil(box.x2 * width))))
    y2 = max(y1 + 1, min(height, int(np.ceil(box.y2 * height))))
    return x1, y1, x2, y2


def _place_badge_for_box(
    *,
    annotation_id: str,
    index: int,
    box: tuple[int, int, int, int],
    badge_w: int,
    badge_h: int,
    gap: int,
    page_width: int,
    page_height: int,
    occupied: list[tuple[int, int, int, int]],
) -> BadgePlacement:
    x1, y1, x2, y2 = box
    page_x_max = max(0, page_width - badge_w)
    page_y_max = max(0, page_height - badge_h)
    candidates: list[tuple[str, int, int, str]] = []
    if y1 >= badge_h + gap:
        candidates.append(("outside_top", min(x1, page_x_max), y1 - badge_h - gap, "x"))
    if x2 + gap + badge_w <= page_width:
        candidates.append(("outside_right", x2 + gap, min(y1, page_y_max), "y"))
    if x1 >= badge_w + gap:
        candidates.append(("outside_left", x1 - badge_w - gap, min(y1, page_y_max), "y"))
    if y2 + gap + badge_h <= page_height:
        candidates.append(("outside_bottom", min(x1, page_x_max), y2 + gap, "x"))
    for placement_name, base_x, base_y, axis in candidates:
        placed = _try_nudged_rect(
            base_x=base_x,
            base_y=base_y,
            badge_w=badge_w,
            badge_h=badge_h,
            page_width=page_width,
            page_height=page_height,
            occupied=occupied,
            axis=axis,
            gap=gap,
        )
        if placed is not None:
            return BadgePlacement(
                annotation_id=annotation_id,
                index=index,
                placement=placement_name,
                x1=placed[0],
                y1=placed[1],
                x2=placed[2],
                y2=placed[3],
            )

    inside_x = min(max(x1 + gap, 0), min(page_x_max, max(x1, x2 - badge_w)))
    inside_y = min(max(y1 + gap, 0), min(page_y_max, max(y1, y2 - badge_h)))
    placed_inside = _try_nudged_rect(
        base_x=inside_x,
        base_y=inside_y,
        badge_w=badge_w,
        badge_h=badge_h,
        page_width=page_width,
        page_height=page_height,
        occupied=occupied,
        axis="x",
        gap=max(2, gap // 2),
    )
    final = placed_inside or (inside_x, inside_y, inside_x + badge_w, inside_y + badge_h)
    return BadgePlacement(
        annotation_id=annotation_id,
        index=index,
        placement="inside",
        x1=final[0],
        y1=final[1],
        x2=final[2],
        y2=final[3],
    )


def _try_nudged_rect(
    *,
    base_x: int,
    base_y: int,
    badge_w: int,
    badge_h: int,
    page_width: int,
    page_height: int,
    occupied: list[tuple[int, int, int, int]],
    axis: str,
    gap: int,
) -> tuple[int, int, int, int] | None:
    collision_pad = max(2, gap // 2)
    step = max(4, gap, min(badge_w, badge_h) // 3)
    if axis == "x":
        max_shift = max(0, page_width - badge_w)
        for offset in _offset_sequence(step, max_shift):
            x1 = base_x + offset
            if x1 < 0 or x1 + badge_w > page_width:
                continue
            rect = (x1, base_y, x1 + badge_w, base_y + badge_h)
            if not _overlaps_any(rect, occupied, padding=collision_pad):
                return rect
        return None
    if axis == "y":
        max_shift = max(0, page_height - badge_h)
        for offset in _offset_sequence(step, max_shift):
            y1 = base_y + offset
            if y1 < 0 or y1 + badge_h > page_height:
                continue
            rect = (base_x, y1, base_x + badge_w, y1 + badge_h)
            if not _overlaps_any(rect, occupied, padding=collision_pad):
                return rect
        return None
    rect = (base_x, base_y, base_x + badge_w, base_y + badge_h)
    if _overlaps_any(rect, occupied, padding=collision_pad):
        return None
    return rect


def _offset_sequence(step: int, limit: int) -> list[int]:
    max_steps = (limit // max(1, step)) + 1
    offsets = [0]
    for idx in range(1, max_steps + 1):
        offsets.append(idx * step)
        offsets.append(-idx * step)
    return offsets


def _overlaps_any(
    rect: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
    *,
    padding: int,
) -> bool:
    return any(_rects_overlap(rect, other, padding=padding) for other in occupied)


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    padding: int = 0,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + padding <= bx1
        or bx2 + padding <= ax1
        or ay2 + padding <= by1
        or by2 + padding <= ay1
    )


def _html_payload(
    manifest: VisualDiffManifest,
    *,
    source_image: BGRImage,
    target_image: BGRImage,
) -> dict[str, object]:
    placements = {
        p.annotation_id: p
        for p in plan_badge_placements_for_canvas(
            manifest.source_image_width,
            manifest.source_image_height,
            manifest.annotations,
        )
    }
    annotations: list[dict[str, object]] = []
    for annotation in manifest.annotations:
        style = _style_for(annotation.match_type)
        crop_data_url = None
        if annotation.target_bbox is not None and annotation.match_type in _TARGET_CROP_MATCH_TYPES:
            crop = _crop_image(target_image, annotation.target_bbox)
            if crop is not None:
                crop_data_url = image_to_data_url(crop)
        placement = placements[annotation.annotation_id]
        annotations.append(
            {
                "annotation_id": annotation.annotation_id,
                "index": annotation.index,
                "match_type": annotation.match_type.value,
                "match_label": style.label,
                "color_hex": style.color_hex,
                "fill_alpha": style.fill_alpha,
                "crop_alpha": style.crop_alpha,
                "source_text": annotation.source_text,
                "target_text": annotation.target_text,
                "confidence": round(annotation.confidence, 4),
                "reason": annotation.reason or "",
                "overlay_bbox_normalized": annotation.overlay_bbox_normalized.model_dump(mode="json"),
                "badge_bbox_normalized": placement.normalized_bbox(
                    page_width=manifest.source_image_width,
                    page_height=manifest.source_image_height,
                ).model_dump(mode="json"),
                "badge_placement": placement.placement,
                "target_crop_data_url": crop_data_url,
            }
        )
    return {
        "manifest": manifest.model_dump(mode="json"),
        "base_image_data_url": image_to_data_url(source_image),
        "annotations": annotations,
    }


def _legend_markup(manifest: VisualDiffManifest) -> str:
    counts = _annotation_counts(manifest.annotations)
    parts: list[str] = []
    for match_type in (
        MatchType.CHANGED_VALUE,
        MatchType.MISSING_IN_TARGET,
        MatchType.EXTRA_IN_TARGET,
        MatchType.UNCERTAIN,
    ):
        style = _style_for(match_type)
        count = counts.get(match_type, 0)
        parts.append(
            "<label>"
            f"<input type=\"checkbox\" data-toggle-type=\"{html.escape(match_type.value)}\" checked />"
            f"<span class=\"swatch\" style=\"background:{html.escape(style.color_hex)}\"></span>"
            f"<span>{html.escape(style.label)} ({count})</span>"
            "</label>"
        )
    return "".join(parts)


def _table_rows_markup(manifest: VisualDiffManifest) -> str:
    rows: list[str] = []
    for annotation in manifest.annotations:
        style = _style_for(annotation.match_type)
        rows.append(
            "<tr"
            f" data-match-type=\"{html.escape(annotation.match_type.value)}\""
            ">"
            f"<td>{annotation.index}</td>"
            f"<td><span class=\"swatch\" style=\"background:{html.escape(style.color_hex)}\"></span> {html.escape(style.label)}</td>"
            f"<td>{html.escape(annotation.source_text or '—')}</td>"
            f"<td>{html.escape(annotation.target_text or '—')}</td>"
            f"<td>{annotation.confidence:.0%}</td>"
            f"<td>{html.escape((annotation.reason or '').strip() or '—')}</td>"
            "</tr>"
        )
    return "".join(rows)


def _empty_table_markup() -> str:
    return (
        "<div class=\"empty\">"
        "No non-exact findings were generated for this run. The canvas is just the source base image."
        "</div>"
    )


def _annotation_counts(annotations: list[VisualAnnotation]) -> dict[MatchType, int]:
    counts: dict[MatchType, int] = {}
    for annotation in annotations:
        counts[annotation.match_type] = counts.get(annotation.match_type, 0) + 1
    return counts


def _safe_script_json(data: dict[str, object]) -> str:
    raw = json.dumps(data, ensure_ascii=False)
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _style_for(match_type: MatchType) -> _VisualStyle:
    try:
        return _STYLE_BY_MATCH_TYPE[match_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported visual diff match type: {match_type!r}") from exc


def _round_unit(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 6)


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    raw = color_hex.lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"Expected 6-digit color, got {color_hex!r}")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _line_thickness(width: int, height: int) -> int:
    return max(1, int(round(max(width, height) / 900.0)))


def _pdf_line_width(width: int, height: int) -> float:
    return max(1.2, min(4.0, max(width, height) / 700.0))


def _font_scale(width: int, height: int) -> float:
    return max(0.45, max(width, height) / 3200.0)


def _badge_font_size(width: int, height: int) -> float:
    return min(28.0, max(10.0, max(width, height) / 70.0))


def _badge_gap(width: int, height: int) -> int:
    return max(4, int(round(_badge_font_size(width, height) * 0.45)))


def _badge_dimensions(text: str, width: int, height: int) -> tuple[int, int]:
    font_px = max(10, int(round(_badge_font_size(width, height))))
    pad_x = max(6, int(round(font_px * 0.45)))
    pad_y = max(4, int(round(font_px * 0.30)))
    text_w = max(font_px, int(math.ceil(font_px * 0.7 * max(1, len(text)))))
    badge_w = text_w + pad_x * 2
    badge_h = font_px + pad_y * 2
    return badge_w, badge_h
