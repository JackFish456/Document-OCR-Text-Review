"""Write error analysis to JSON / Markdown and print a console summary."""

from __future__ import annotations

import json
from pathlib import Path

from app.models.error_analysis import ErrorAnalysisReport, FieldFailureRecord


def write_error_analysis_json(
    report: ErrorAnalysisReport,
    path: Path,
    *,
    max_failures_in_json: int = 500,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report.model_dump(mode="json")
    failures = data.get("failures", [])
    if max_failures_in_json >= 0 and len(failures) > max_failures_in_json:
        data["failures"] = failures[:max_failures_in_json]
        data["failures_omitted_count"] = len(failures) - max_failures_in_json
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fmt_example(f: FieldFailureRecord, idx: int) -> list[str]:
    lines = [
        f"{idx}. **{f.pair_id}** / `{f.field_id}` — `{f.failure_pattern}`",
        f"   - Gold: `{f.gold_expected.value}`  |  Pred: `{f.predicted_match_type or '∅'}` "
        f"(bucket `{f.predicted_bucket}`)  conf={f.match_confidence}",
    ]
    if f.source_value_preview or f.target_value_preview:
        lines.append(
            f"   - Values: src={f.source_value_preview!r}  tgt={f.target_value_preview!r}"
        )
    if f.min_field_confidence is not None:
        lines.append(f"   - Min field OCR/confidence: {f.min_field_confidence:.3f}")
    if f.engine_reason:
        lines.append(f"   - Engine: {f.engine_reason}")
    lines.append(f"   - *{f.rationale}*")
    return lines


def build_error_analysis_markdown(report: ErrorAnalysisReport) -> str:
    lines: list[str] = [
        "# Error analysis report",
        "",
        "## Run",
        "",
        f"- **run_id:** `{report.run_id}`",
        f"- **manifest:** `{report.manifest_name}`",
        f"- **OCR provider:** `{report.ocr_provider or 'default'}`",
        f"- **OCR confidence threshold (heuristic):** `{report.ocr_confidence_threshold}`",
        f"- **project_root:** `{report.project_root}`",
        "",
        "## Coverage",
        "",
        f"- Manifest rows skipped (no annotation): **{report.pairs_manifest_no_annotation}**",
        f"- Pipeline failures: **{report.pairs_pipeline_failed}**",
        f"- Pairs scored: **{report.pairs_succeeded}**",
        f"- Total gold fields: **{report.total_gold_fields}**",
        f"- Field-level failures: **{report.total_failures}**",
        f"- Pairs with ≥1 field failure: **{report.pairs_with_failures}**",
        "",
    ]
    if report.pipeline_failures:
        lines.extend(["## Pipeline failures", ""])
        for p in report.pipeline_failures[:25]:
            lines.append(f"- `{p.pair_id}`: {p.error_message}")
        if len(report.pipeline_failures) > 25:
            lines.append(f"- … ({len(report.pipeline_failures) - 25} more)")
        lines.append("")

    lines.extend(
        [
            "## Failures by category",
            "",
            "| Category | Count |",
            "|----------|-------:|",
        ]
    )
    for cat in ("ocr", "extraction", "matching", "classification"):
        n = report.failures_by_category.get(cat, 0)
        lines.append(f"| {cat} | {n} |")
    lines.append("")

    lines.extend(["## Most common failure patterns", "", "| Pattern | Count |", "|---------|-------:|"])
    for pat, cnt in report.top_failure_patterns:
        lines.append(f"| `{pat}` | {cnt} |")
    lines.append("")

    for cat in ("ocr", "extraction", "matching", "classification"):
        ex = report.examples_by_category.get(cat, [])
        if not ex:
            continue
        title = cat.replace("_", " ").title()
        lines.extend([f"## Example failures — {title}", ""])
        for i, f in enumerate(ex, start=1):
            lines.extend(_fmt_example(f, i))
            lines.append("")

    lines.append(
        "---\n\n*Categories: **ocr** = low field confidence on a pair; **extraction** = gold "
        "field_id missing from parsed fields / no match row; **matching** = wrong alignment or "
        "uncertain pair; **classification** = pair aligned but MatchType disagrees with gold.*"
    )
    return "\n".join(lines)


def write_error_analysis_markdown(report: ErrorAnalysisReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_error_analysis_markdown(report), encoding="utf-8")


def print_error_analysis_console(report: ErrorAnalysisReport) -> None:
    text = build_error_analysis_markdown(report)
    # Console: strip markdown emphasis for simpler terminals
    plain = text.replace("**", "").replace("`", "")
    print(plain)
