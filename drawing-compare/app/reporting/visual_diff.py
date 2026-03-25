# ruff: noqa: E501
"""Visual overlay diff artifacts for local drawing review."""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from app.models.comparison import CompareResponse
from app.models.match import ComparisonSummary, MatchResult, MatchType
from app.models.ocr import BoundingBox
from app.preprocessing.types import BGRImage
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


class VisualDiffManifest(BaseModel):
    """JSON-serializable manifest for visual diff artifacts."""

    schema_version: str = "1.0"
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

    @model_validator(mode="after")
    def _sync_annotation_count(self) -> VisualDiffManifest:
        self.annotation_count = len(self.annotations)
        return self


@dataclass(slots=True)
class VisualDiffArtifacts:
    """Rendered outputs for local/manual review runs."""

    manifest: VisualDiffManifest
    overlay_bgr: BGRImage
    html: str


@dataclass(frozen=True, slots=True)
class _VisualStyle:
    label: str
    color_hex: str
    fill_alpha: float
    crop_alpha: float

    @property
    def rgb(self) -> tuple[int, int, int]:
        return _hex_to_rgb(self.color_hex)

    @property
    def bgr(self) -> tuple[int, int, int]:
        r, g, b = self.rgb
        return b, g, r


_STYLE_BY_MATCH_TYPE: dict[MatchType, _VisualStyle] = {
    MatchType.CHANGED_VALUE: _VisualStyle("Changed text", "#d97706", 0.12, 0.48),
    MatchType.MISSING_IN_TARGET: _VisualStyle("Missing on target", "#b42318", 0.26, 0.0),
    MatchType.EXTRA_IN_TARGET: _VisualStyle("Only on target", "#0f766e", 0.12, 0.44),
    MatchType.UNCERTAIN: _VisualStyle("Unclear match", "#7c6f10", 0.18, 0.34),
}


def build_visual_diff_artifacts(
    response: CompareResponse,
    compare_artifacts: CompareArtifacts,
) -> VisualDiffArtifacts:
    """Build manifest, flattened PNG image, and self-contained HTML report."""
    manifest = build_visual_diff_manifest(response, compare_artifacts)
    overlay_bgr = render_visual_diff_overlay(
        manifest,
        source_image=compare_artifacts.source_page.processed_bgr,
        target_image=compare_artifacts.target_page.processed_bgr,
    )
    html_report = build_visual_diff_html(
        manifest,
        source_image=compare_artifacts.source_page.processed_bgr,
        target_image=compare_artifacts.target_page.processed_bgr,
    )
    return VisualDiffArtifacts(manifest=manifest, overlay_bgr=overlay_bgr, html=html_report)


def build_visual_diff_manifest(
    response: CompareResponse,
    compare_artifacts: CompareArtifacts,
) -> VisualDiffManifest:
    """Convert non-exact `MatchResult` rows into a stable visual manifest."""
    source_h, source_w = compare_artifacts.source_page.processed_bgr.shape[:2]
    target_h, target_w = compare_artifacts.target_page.processed_bgr.shape[:2]
    annotations: list[VisualAnnotation] = []
    for idx, result in enumerate(_iter_visual_results(compare_artifacts.results), start=1):
        annotations.append(
            _build_annotation(
                result,
                index=idx,
                source_width=source_w,
                source_height=source_h,
                target_width=target_w,
                target_height=target_h,
            )
        )
    return VisualDiffManifest(
        comparison_id=str(response.comparison_id),
        source_path=str(response.extras.get("source_path", "")),
        target_path=str(response.extras.get("target_path", "")),
        ocr_provider=str(response.extras.get("ocr_provider", "")),
        source_page_number=compare_artifacts.source_page.page_number,
        target_page_number=compare_artifacts.target_page.page_number,
        source_image_width=source_w,
        source_image_height=source_h,
        target_image_width=target_w,
        target_image_height=target_h,
        summary=response.report.summary,
        annotations=annotations,
    )


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
            x1=x1,
            y1=y1,
            bgr=style.bgr,
            font_scale=font_scale,
            thickness=thickness,
        )
    return canvas


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

    function drawBadge(x, y, text, color) {
      const fontPx = Math.max(14, Math.round(Math.max(canvas.width, canvas.height) / 140));
      ctx.save();
      ctx.font = `700 ${fontPx}px Bahnschrift, Trebuchet MS, sans-serif`;
      const textWidth = ctx.measureText(text).width;
      const badgeW = Math.ceil(textWidth) + 16;
      const badgeH = fontPx + 10;
      const badgeY = Math.max(0, y - badgeH);
      ctx.fillStyle = color;
      ctx.fillRect(x, badgeY, badgeW, badgeH);
      ctx.fillStyle = "#ffffff";
      ctx.textBaseline = "top";
      ctx.fillText(text, x + 8, badgeY + 5);
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
          drawBadge(x, y, String(ann.index), ann.color_hex);
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
    x1: int,
    y1: int,
    bgr: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    pad = max(4, thickness + 1)
    badge_w = tw + pad * 2
    badge_h = th + pad * 2 + baseline
    badge_y1 = max(0, y1 - badge_h)
    cv2.rectangle(
        image,
        (x1, badge_y1),
        (min(image.shape[1], x1 + badge_w), min(image.shape[0], badge_y1 + badge_h)),
        bgr,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x1 + pad, badge_y1 + pad + th),
        font,
        font_scale,
        (255, 255, 255),
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
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


def _html_payload(
    manifest: VisualDiffManifest,
    *,
    source_image: BGRImage,
    target_image: BGRImage,
) -> dict[str, object]:
    annotations: list[dict[str, object]] = []
    for annotation in manifest.annotations:
        style = _style_for(annotation.match_type)
        crop_data_url = None
        if annotation.target_bbox is not None and annotation.match_type in _TARGET_CROP_MATCH_TYPES:
            crop = _crop_image(target_image, annotation.target_bbox)
            if crop is not None:
                crop_data_url = image_to_data_url(crop)
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


def _font_scale(width: int, height: int) -> float:
    return max(0.45, max(width, height) / 3200.0)
