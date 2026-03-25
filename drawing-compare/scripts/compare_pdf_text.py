#!/usr/bin/env python3
"""Compare embedded PDF text and emit readable Markdown/HTML reports."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import fitz
from rapidfuzz import fuzz


# Meaningful short / drawing tokens we never treat as noise.
_MEANINGFUL_SHORT = re.compile(
    r"(?ix)^("
    r"[a-z]\d{0,3}|"  # grid: E, A1
    r"\d{1,2}\s*['\u2019\u2032]\s*[-–—]?\s*\d{1,2}\s*['\u2019\u2032\"\u201d\u2033]?|"  # 0' - 6"
    r"\d+\s*['\u2019\u2032]\s*\d+\s*['\u2019\u2032\"\u201d\u2033]?|"
    r"\d+\s*['\u2019\u2032\"\u201d\u2033]|"  # 6"
    r"\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4}|"  # dates
    r"\d+(\.\d+)?\s*[x×]\s*\d+(\.\d+)?|"  # 12x24
    r"[\d.\s'\"°\-–—/]{2,}"  # mostly numeric dimension fragments
    r")$"
)

# Replacement char or high private-use-area density (common OCR garbage).
_PRIVATE_USE = re.compile(r"[\uE000-\uF8FF]")

ChangeCategory = Literal[
    "dimension",
    "date",
    "revision_status",
    "identifier_code",
    "generic_text",
]

_RECAP_TOP_CHANGES = 8
_RECAP_NOISE_MAX = 5
_SNIPPET_MAX = 72
_TOP_CHANGED_SNIPPETS = 5
_TOP_MISSING_SNIPPETS = 3
_TOP_EXTRA_SNIPPETS = 3

_AI_MAX_BULLETS = 8
_AI_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_AI_TIMEOUT_SEC = 90
_NUM_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")

# Heuristics for change type (first match wins; order = most specific first).
_RE_DIMENSION = re.compile(
    r"(?ix)"
    r"\d+\s*['\u2019\u2032]|"
    r'\d+\s*["\u201d\u2033]|'
    r"\d+(\.\d+)?\s*[x×]\s*\d+(\.\d+)?|"
    r"\b\d+\s*[-–—]\s*\d+\s*['\u2019\u2032]|"
    r"\b\d+(\.\d+)?\s*(mm|cm|m|ft|in|inch|inches|deg|degrees|dia|diameter|radius)\b|"
    r"\b(elevation|el\.|slope|grade)\b.*\d|"
    r"^\s*[\d.\s'\"°\-–—/]{2,}\s*$"
)
_RE_DATE = re.compile(
    r"(?ix)"
    r"\b\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4}\b|"
    r"\b\d{4}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{1,2}\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(,?\s*\d{4})?\b"
)
_RE_REVISION = re.compile(
    r"(?i)"
    r"\b(rev|revision|revisions?|status|ifc|fcd|afc|for\s+construction|"
    r"as[\s-]+built|issued|issue\s*#|construction\s+admin)\b|"
    r"\b(ifc|fcd|afc)\b|"
    r"\brev\s*[.:]?\s*[a-z0-9]{1,6}\b"
)
_RE_IDENTIFIER = re.compile(
    r"(?i)"
    r"\b[A-Z]{2,12}[-_/]\d{2,8}\b|"
    r"\b[A-Z]{2,12}\s*#\s*\d+\b|"
    r"\b(dwg|drawing|sheet|sht|detail|spec)\s*[#:]?\s*[\w.-]+\b|"
    r"\b\d{3,}[-_]\d{2,}\b"
)

_CATEGORY_ORDER: dict[ChangeCategory, int] = {
    "dimension": 0,
    "date": 1,
    "revision_status": 2,
    "identifier_code": 3,
    "generic_text": 4,
}


@dataclass(frozen=True)
class TextLine:
    page: int
    text: str
    normalized: str
    orientation_deg: float
    is_vertical: bool


@dataclass
class PageExtractStats:
    raw_lines: int = 0
    noise_dropped: int = 0
    vertical_skipped: int = 0
    vertical_kept: int = 0
    dedupe_removed: int = 0


@dataclass
class DocumentExtractStats:
    pages: dict[int, PageExtractStats] = field(default_factory=dict)

    def page(self, p: int) -> PageExtractStats:
        if p not in self.pages:
            self.pages[p] = PageExtractStats()
        return self.pages[p]

    @property
    def raw_lines(self) -> int:
        return sum(ps.raw_lines for ps in self.pages.values())

    @property
    def noise_dropped(self) -> int:
        return sum(ps.noise_dropped for ps in self.pages.values())

    @property
    def vertical_skipped(self) -> int:
        return sum(ps.vertical_skipped for ps in self.pages.values())

    @property
    def vertical_kept(self) -> int:
        return sum(ps.vertical_kept for ps in self.pages.values())

    @property
    def dedupe_removed(self) -> int:
        return sum(ps.dedupe_removed for ps in self.pages.values())

    @property
    def filtered_lines(self) -> int:
        return self.raw_lines - self.noise_dropped - self.vertical_skipped


@dataclass(frozen=True)
class PairScore:
    src_i: int
    tgt_i: int
    score: float


def _line_orientation_deg(dir_vec: object) -> float:
    if not dir_vec or not isinstance(dir_vec, (tuple, list)) or len(dir_vec) < 2:
        return 0.0
    try:
        dx, dy = float(dir_vec[0]), float(dir_vec[1])
    except (TypeError, ValueError):
        return 0.0
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _is_vertical_orientation(deg: float, tol: float = 15.0) -> bool:
    a = deg % 360.0
    if a > 180.0:
        a -= 360.0
    return min(abs(a - 90.0), abs(a + 90.0)) <= tol


def _norm(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s


def _meaningful_token(t: str) -> bool:
    if _MEANINGFUL_SHORT.match(t.strip()):
        return True
    if len(t) <= 4 and re.search(r"[a-z0-9]", t, re.I):
        return True
    return False


def _symbol_noise_ratio(t: str) -> float:
    if not t:
        return 1.0
    sym = sum(1 for c in t if not (c.isalnum() or c.isspace() or c in "\"'°.-–—/\\"))
    return sym / len(t)


def _is_noise_line(text: str, strictness: float) -> bool:
    """Drop OCR-like garbage; strictness in [0,1] raises the bar when higher."""
    t = text.strip()
    if not t:
        return True
    if _meaningful_token(t):
        return False
    if "\ufffd" in t:
        return True
    pua = len(_PRIVATE_USE.findall(t))
    if len(t) >= 3 and pua / len(t) >= 0.2 + 0.5 * strictness:
        return True

    alnum = sum(1 for c in t if c.isalnum())
    alnum_ratio = alnum / max(len(t), 1)
    # Higher strictness → require cleaner lines for non-trivial length.
    min_alnum = 0.12 + 0.48 * strictness
    if len(t) >= 5 and alnum_ratio < min_alnum:
        return True

    sym_r = _symbol_noise_ratio(t)
    sym_thresh = 0.55 + 0.35 * strictness
    if len(t) >= 4 and sym_r > sym_thresh:
        return True

    # Long runs of a single character (e.g. decorative rules mistaken as text).
    compact = t.replace(" ", "")
    if len(compact) >= 6:
        most = max(compact.count(c) for c in set(compact))
        if most / len(compact) >= 0.88 - 0.08 * strictness:
            return True

    return False


def _join_line_spans(spans: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for sp in spans:
        parts.append(str(sp.get("text", "")))
    return " ".join(s for s in parts if s).strip()


def extract_pdf_lines(
    path: Path,
    *,
    strictness: float,
    disable_vertical: bool,
) -> tuple[list[TextLine], DocumentExtractStats]:
    doc = fitz.open(path)
    stats = DocumentExtractStats()
    try:
        pre_dedupe: list[TextLine] = []
        for pidx in range(len(doc)):
            page = doc.load_page(pidx)
            page_no = pidx + 1
            ps = stats.page(page_no)
            pd = page.get_text("dict")
            blocks = pd.get("blocks", [])
            for b in blocks:
                lines = b.get("lines", [])
                for ln in lines:
                    if not isinstance(ln, dict):
                        continue
                    spans = ln.get("spans", [])
                    if not isinstance(spans, list):
                        continue
                    raw = _join_line_spans(spans)
                    if not raw:
                        continue
                    ps.raw_lines += 1
                    orient = _line_orientation_deg(ln.get("dir"))
                    vert = _is_vertical_orientation(orient)

                    if disable_vertical and vert:
                        ps.vertical_skipped += 1
                        continue

                    if _is_noise_line(raw, strictness):
                        ps.noise_dropped += 1
                        continue

                    if vert:
                        ps.vertical_kept += 1

                    n = _norm(raw)
                    if not n:
                        ps.noise_dropped += 1
                        continue
                    pre_dedupe.append(
                        TextLine(
                            page=page_no,
                            text=raw,
                            normalized=n,
                            orientation_deg=orient,
                            is_vertical=vert,
                        )
                    )

        # Dedupe identical normalized text on the same page (preserve first occurrence).
        seen: set[tuple[int, str]] = set()
        out: list[TextLine] = []
        for tl in pre_dedupe:
            key = (tl.page, tl.normalized)
            if key in seen:
                stats.page(tl.page).dedupe_removed += 1
                continue
            seen.add(key)
            out.append(tl)
        return out, stats
    finally:
        doc.close()


def default_match_threshold(strictness: float) -> float:
    """Higher strictness → stricter 'unchanged' classification (fuzz ratio)."""
    return 80.0 + 13.0 * strictness


def default_min_candidate(strictness: float) -> float:
    """Higher strictness → higher bar to form a candidate pair (still pairs small dim/date edits)."""
    return 50.0 + 22.0 * strictness


def match_lines(
    source: list[TextLine],
    target: list[TextLine],
    *,
    min_candidate: float,
) -> tuple[list[PairScore], list[int], list[int]]:
    candidates: list[PairScore] = []
    for i, s in enumerate(source):
        for j, t in enumerate(target):
            score = float(fuzz.ratio(s.normalized, t.normalized))
            if score >= min_candidate:
                candidates.append(PairScore(src_i=i, tgt_i=j, score=score))
    candidates.sort(key=lambda c: c.score, reverse=True)

    used_src: set[int] = set()
    used_tgt: set[int] = set()
    pairs: list[PairScore] = []
    for c in candidates:
        if c.src_i in used_src or c.tgt_i in used_tgt:
            continue
        used_src.add(c.src_i)
        used_tgt.add(c.tgt_i)
        pairs.append(c)

    missing_src = [i for i in range(len(source)) if i not in used_src]
    extra_tgt = [j for j in range(len(target)) if j not in used_tgt]
    return pairs, missing_src, extra_tgt


def classify_change_category(src: str, tgt: str) -> ChangeCategory:
    blob = f"{src} {tgt}".strip()
    if not blob:
        return "generic_text"
    if _RE_DIMENSION.search(blob):
        return "dimension"
    if _RE_DATE.search(blob):
        return "date"
    if _RE_REVISION.search(blob):
        return "revision_status"
    if _RE_IDENTIFIER.search(blob):
        return "identifier_code"
    return "generic_text"


def _digit_groups(s: str) -> list[str]:
    return re.findall(r"\d+", s)


def change_risk_score(
    category: ChangeCategory,
    src: str,
    tgt: str,
    *,
    fuzz_score: float | None,
    match_threshold: float,
) -> int:
    """0-100: higher = more likely a substantive document change worth review."""
    base = {
        "dimension": 78,
        "date": 82,
        "revision_status": 70,
        "identifier_code": 66,
        "generic_text": 36,
    }[category]
    ds, dt = _digit_groups(src), _digit_groups(tgt)
    if ds != dt and (ds or dt):
        base += 14
    if fuzz_score is not None:
        gap = max(0.0, match_threshold - float(fuzz_score))
        base += int(min(18.0, gap * 0.65))
    if len(src.strip()) <= 2 or len(tgt.strip()) <= 2:
        base = max(base - 8, 0)
    return int(min(100, base))


def extraction_noise_likelihood(text: str) -> int:
    """0-100: higher = more likely PDF extraction/OCR garbage (not authoritative text)."""
    t = text.strip()
    if not t:
        return 0
    score = 0
    if "\ufffd" in t:
        score += 45
    pua = len(_PRIVATE_USE.findall(t))
    if len(t) >= 2:
        score += int(min(40, (pua / len(t)) * 120))
    ar = sum(1 for c in t if c.isalnum()) / len(t)
    if len(t) >= 4 and ar < 0.35:
        score += 28
    sym = _symbol_noise_ratio(t)
    if len(t) >= 3 and sym > 0.45:
        score += 22
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 5:
        uniq = len(set(compact))
        if uniq <= 2:
            score += 25
    if re.fullmatch(r"[\W_]+", t):
        score += 50
    return int(min(100, score))


def _snippet(text: str, max_len: int = _SNIPPET_MAX) -> str:
    s = text.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "..."


def _recap_sort_key(
    risk: int,
    category: ChangeCategory,
    fuzz: float | None,
) -> tuple[int, int, float]:
    cat_pri = _CATEGORY_ORDER[category]
    fuzz_key = -(fuzz or 0.0)
    return (-risk, cat_pri, fuzz_key)


def build_recap_payload(
    changed: list[dict[str, object]],
    missing: list[dict[str, object]],
    extra: list[dict[str, object]],
    *,
    match_threshold: float,
    min_candidate: float,
) -> dict[str, object]:
    """Structured recap for JSON and renderers; keeps lists short."""
    cat_changed: dict[str, int] = {k: 0 for k in _CATEGORY_ORDER}
    for r in changed:
        c = str(r["category"])
        if c in cat_changed:
            cat_changed[c] += 1

    candidates: list[dict[str, object]] = []
    for r in changed:
        candidates.append(
            {
                "kind": "changed",
                "risk_score": r["risk_score"],
                "category": r["category"],
                "fuzz_score": r["score"],
                "source_page": r["source_page"],
                "target_page": r["target_page"],
                "source_snippet": _snippet(str(r["source_text"])),
                "target_snippet": _snippet(str(r["target_text"])),
            }
        )
    for r in missing:
        candidates.append(
            {
                "kind": "missing",
                "risk_score": r["risk_score"],
                "category": r["category"],
                "fuzz_score": None,
                "source_page": r["source_page"],
                "target_page": None,
                "source_snippet": _snippet(str(r["source_text"])),
                "target_snippet": "",
            }
        )
    for r in extra:
        candidates.append(
            {
                "kind": "extra",
                "risk_score": r["risk_score"],
                "category": r["category"],
                "fuzz_score": None,
                "source_page": None,
                "target_page": r["target_page"],
                "source_snippet": "",
                "target_snippet": _snippet(str(r["target_text"])),
            }
        )

    candidates.sort(
        key=lambda x: _recap_sort_key(
            int(x["risk_score"]),
            x["category"],  # type: ignore[arg-type]
            float(x["fuzz_score"]) if x["fuzz_score"] is not None else None,
        )
    )
    top_changes = candidates[:_RECAP_TOP_CHANGES]

    noise_scored: list[tuple[int, str, dict[str, object]]] = []
    for r in changed:
        src, tgt = str(r["source_text"]), str(r["target_text"])
        n = max(extraction_noise_likelihood(src), extraction_noise_likelihood(tgt))
        fs = float(r["score"])
        if fs <= min_candidate + 3.0:
            n = min(100, n + 12)
        if str(r["category"]) == "generic_text" and int(r["risk_score"]) < 48:
            n = min(100, n + 8)
        noise_scored.append((n, "changed", r))
    for r in missing:
        noise_scored.append((extraction_noise_likelihood(str(r["source_text"])), "missing", r))
    for r in extra:
        noise_scored.append((extraction_noise_likelihood(str(r["target_text"])), "extra", r))
    noise_scored.sort(key=lambda t: -t[0])
    noise_bullets: list[dict[str, object]] = []
    seen_noise_key: set[tuple[str, str, str]] = set()
    for nscore, nk, rr in noise_scored:
        if nscore < 25 and len(noise_bullets) >= 2:
            continue
        if nk == "changed":
            pages = f"p{rr['source_page']}/p{rr['target_page']}"
            text = f"{_snippet(str(rr['source_text']))} -> {_snippet(str(rr['target_text']))}"
            key = ("c", str(rr["source_text"]), str(rr["target_text"]))
        elif nk == "missing":
            pages = f"p{rr['source_page']}"
            text = _snippet(str(rr["source_text"]))
            key = ("m", str(rr["source_text"]), "")
        else:
            pages = f"p{rr['target_page']}"
            text = _snippet(str(rr["target_text"]))
            key = ("e", str(rr["target_text"]), "")
        if key in seen_noise_key:
            continue
        seen_noise_key.add(key)
        noise_bullets.append({"noise_score": nscore, "kind": nk, "pages": pages, "text": text})
        if len(noise_bullets) >= _RECAP_NOISE_MAX:
            break

    ch_sorted = sorted(
        changed,
        key=lambda r: _recap_sort_key(
            int(r["risk_score"]),
            r["category"],  # type: ignore[arg-type]
            float(r["score"]),
        ),
    )
    miss_sorted = sorted(
        missing,
        key=lambda r: (-int(r["risk_score"]), _CATEGORY_ORDER[r["category"]]),  # type: ignore[arg-type]
    )
    extra_sorted = sorted(
        extra,
        key=lambda r: (-int(r["risk_score"]), _CATEGORY_ORDER[r["category"]]),  # type: ignore[arg-type]
    )

    high_risk_changed = sum(1 for r in changed if int(r["risk_score"]) >= 72)

    return {
        "summary_counts": {
            "changed_total": len(changed),
            "missing_total": len(missing),
            "extra_total": len(extra),
            "changed_by_category": cat_changed,
            "high_risk_changed": high_risk_changed,
        },
        "top_potential_changes": top_changes,
        "likely_extraction_noise": noise_bullets,
        "top_snippets": {
            "changed": [
                {
                    "risk_score": r["risk_score"],
                    "category": r["category"],
                    "score": r["score"],
                    "source_page": r["source_page"],
                    "target_page": r["target_page"],
                    "source_snippet": _snippet(str(r["source_text"])),
                    "target_snippet": _snippet(str(r["target_text"])),
                }
                for r in ch_sorted[:_TOP_CHANGED_SNIPPETS]
            ],
            "missing": [
                {
                    "risk_score": r["risk_score"],
                    "category": r["category"],
                    "source_page": r["source_page"],
                    "snippet": _snippet(str(r["source_text"])),
                }
                for r in miss_sorted[:_TOP_MISSING_SNIPPETS]
            ],
            "extra": [
                {
                    "risk_score": r["risk_score"],
                    "category": r["category"],
                    "target_page": r["target_page"],
                    "snippet": _snippet(str(r["target_text"])),
                }
                for r in extra_sorted[:_TOP_EXTRA_SNIPPETS]
            ],
        },
    }


def _stats_blob(st: DocumentExtractStats) -> dict[str, int]:
    return {
        "raw_lines": st.raw_lines,
        "filtered_lines": st.filtered_lines,
        "noise_dropped": st.noise_dropped,
        "vertical_skipped": st.vertical_skipped,
        "vertical_kept": st.vertical_kept,
        "dedupe_removed": st.dedupe_removed,
        "lines_after_dedupe": st.filtered_lines - st.dedupe_removed,
    }


def build_report(
    source: list[TextLine],
    target: list[TextLine],
    pairs: list[PairScore],
    missing_src: list[int],
    extra_tgt: list[int],
    *,
    match_threshold: float,
    min_candidate: float,
    source_stats: DocumentExtractStats,
    target_stats: DocumentExtractStats,
    run_options: dict[str, object],
) -> dict[str, object]:
    matched: list[dict[str, object]] = []
    changed: list[dict[str, object]] = []
    for p in pairs:
        s = source[p.src_i]
        t = target[p.tgt_i]
        cat = classify_change_category(s.text, t.text)
        risk = change_risk_score(cat, s.text, t.text, fuzz_score=p.score, match_threshold=match_threshold)
        row = {
            "score": round(p.score, 2),
            "source_page": s.page,
            "target_page": t.page,
            "source_text": s.text,
            "target_text": t.text,
            "source_vertical": s.is_vertical,
            "target_vertical": t.is_vertical,
            "category": cat,
            "risk_score": risk,
        }
        if p.score >= match_threshold:
            matched.append(row)
        else:
            changed.append(row)

    missing_rows: list[dict[str, object]] = []
    for i in missing_src:
        s = source[i]
        cat = classify_change_category(s.text, "")
        risk = change_risk_score(cat, s.text, "", fuzz_score=None, match_threshold=match_threshold)
        if cat in ("dimension", "date", "revision_status"):
            risk = min(100, risk + 6)
        missing_rows.append(
            {
                "source_page": s.page,
                "source_text": s.text,
                "source_vertical": s.is_vertical,
                "category": cat,
                "risk_score": risk,
            }
        )

    extra_rows: list[dict[str, object]] = []
    for j in extra_tgt:
        t = target[j]
        cat = classify_change_category("", t.text)
        risk = change_risk_score(cat, "", t.text, fuzz_score=None, match_threshold=match_threshold)
        if cat in ("dimension", "date", "revision_status"):
            risk = min(100, risk + 6)
        extra_rows.append(
            {
                "target_page": t.page,
                "target_text": t.text,
                "target_vertical": t.is_vertical,
                "category": cat,
                "risk_score": risk,
            }
        )

    recap = build_recap_payload(
        changed,
        missing_rows,
        extra_rows,
        match_threshold=match_threshold,
        min_candidate=min_candidate,
    )

    return {
        "options": run_options,
        "extraction": {
            "source": _stats_blob(source_stats),
            "target": _stats_blob(target_stats),
        },
        "summary": {
            "source_lines": len(source),
            "target_lines": len(target),
            "matched": len(matched),
            "changed": len(changed),
            "missing": len(missing_rows),
            "extra": len(extra_rows),
        },
        "recap": recap,
        "matched": matched,
        "changed": changed,
        "missing": missing_rows,
        "extra": extra_rows,
    }


def build_ai_grounding_payload(report: dict[str, object]) -> dict[str, object]:
    """Compact JSON for the model and for numeric grounding (no full diff dump)."""
    recap = report["recap"]  # type: ignore[index]
    opts = report["options"]  # type: ignore[index]
    return {
        "meta": {
            "max_bullets": _AI_MAX_BULLETS,
            "high_risk_changed_threshold": 72,
            "risk_score_max": 100,
            "note": "Cite only numbers that appear in this JSON (including snippet strings).",
        },
        "summary": report["summary"],
        "recap_summary_counts": recap["summary_counts"],
        "top_potential_changes": recap["top_potential_changes"],
        "thresholds_from_run": {
            "strictness": opts["strictness"],
            "match_threshold": opts["match_threshold"],
            "min_candidate": opts["min_candidate"],
        },
    }


def grounding_numbers_from_object(obj: object) -> set[str]:
    """Digit tokens allowed in assistant text (must appear in serialized grounding payload)."""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return set(_NUM_TOKEN_RE.findall(raw))


def ai_violating_number_tokens(text: str, allowed: set[str]) -> list[str]:
    found = set(_NUM_TOKEN_RE.findall(text))
    return sorted(t for t in found if t not in allowed)


def _strip_outer_markdown_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if len(lines) < 2:
        return t
    lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_ai_bullet_lines(text: str) -> list[str]:
    bullets: list[str] = []
    for line in text.strip().splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^[-*•]\s+", s):
            bullets.append(re.sub(r"^[-*•]\s+", "", s).strip())
        elif re.match(r"^\d+\.\s+", s):
            bullets.append(re.sub(r"^\d+\.\s+", "", s).strip())
    return bullets


def openai_chat_completion(api_key: str, model: str, system: str, user: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _AI_OPENAI_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_AI_TIMEOUT_SEC) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("choices[0].message.content is not a string")
    return content.strip()


def try_generate_ai_summary(
    report: dict[str, object],
    *,
    enabled: bool,
    model: str,
    api_key: str | None,
) -> dict[str, object]:
    base: dict[str, object] = {
        "requested": enabled,
        "model": model if enabled else None,
    }
    if not enabled:
        return {**base, "status": "not_requested", "bullets": None, "error_message": None}

    if not api_key or not str(api_key).strip():
        return {
            **base,
            "status": "skipped_no_key",
            "bullets": None,
            "error_message": "OPENAI_API_KEY is not set or empty.",
        }

    payload = build_ai_grounding_payload(report)
    allowed_nums = grounding_numbers_from_object(payload)

    system = (
        "You summarize PDF embedded-text comparison results for construction or engineering drawings. "
        "Use ONLY facts supported by the user JSON. Write at most 8 bullet lines. "
        "Each bullet must start with '- ' (dash and space). No title, no preamble, no closing paragraph. "
        "Prioritize items with kind 'changed' where category is dimension, date, or revision_status, "
        "and items with the highest risk_score. "
        "You may quote short snippet text exactly as it appears in the JSON. "
        "Do not state totals, counts, or numeric metrics unless those exact numbers appear in the JSON "
        "(including meta and thresholds_from_run). Never invent page numbers, dimensions, or dates."
    )
    user = (
        "Summarize the highest-impact deltas for a reviewer. Output markdown bullets only.\n\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n```"
    )

    try:
        raw = openai_chat_completion(str(api_key).strip(), model, system, user)
    except urllib.error.HTTPError as e:
        err_txt = e.read().decode("utf-8", errors="replace")[:2000]
        return {
            **base,
            "status": "error",
            "bullets": None,
            "error_message": f"HTTP {e.code}: {err_txt}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {**base, "status": "error", "bullets": None, "error_message": str(e)}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        return {**base, "status": "error", "bullets": None, "error_message": f"Invalid API response: {e}"}

    raw = _strip_outer_markdown_fence(raw)
    violations = ai_violating_number_tokens(raw, allowed_nums)
    if violations:
        return {
            **base,
            "status": "grounding_failed",
            "bullets": None,
            "error_message": "Assistant used numeric tokens not present in grounding JSON.",
            "grounding_violations": violations,
            "rejected_raw_char_count": len(raw),
        }

    bullets = parse_ai_bullet_lines(raw)
    if not bullets:
        return {
            **base,
            "status": "format_failed",
            "bullets": None,
            "error_message": "Assistant response contained no bullet lines starting with '- '.",
            "rejected_raw_char_count": len(raw),
        }
    if len(bullets) > _AI_MAX_BULLETS:
        return {
            **base,
            "status": "format_failed",
            "bullets": None,
            "error_message": f"Expected at most {_AI_MAX_BULLETS} bullets, got {len(bullets)}.",
            "rejected_raw_char_count": len(raw),
        }

    return {
        **base,
        "status": "ok",
        "bullets": bullets,
        "error_message": None,
        "bullet_count": len(bullets),
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None._"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _markdown_recap_section(recap: dict[str, object]) -> list[str]:
    lines: list[str] = []
    sc = recap["summary_counts"]  # type: ignore[index]
    cbc = sc["changed_by_category"]  # type: ignore[index]
    cat_bits = ", ".join(f"{k}={cbc[k]}" for k in _CATEGORY_ORDER if int(cbc[k]) > 0)
    if not cat_bits:
        cat_bits = "(none)"

    lines.append("## Recap (potential issues)")
    lines.append("")
    lines.append("### Summary counts")
    lines.append("")
    lines.append(
        _md_table(
            ["Metric", "Count"],
            [
                ["Changed lines (paired, below match threshold)", str(sc["changed_total"])],  # type: ignore[index]
                ["Missing on target (unpaired source)", str(sc["missing_total"])],  # type: ignore[index]
                ["Extra on target (unpaired target)", str(sc["extra_total"])],  # type: ignore[index]
                ["High-risk changed (risk >= 72)", str(sc["high_risk_changed"])],  # type: ignore[index]
            ],
        )
    )
    lines.append("")
    lines.append(f"**Changed by category:** {cat_bits}")
    lines.append("")
    lines.append("### Top potential changes")
    lines.append("")
    top = recap["top_potential_changes"]  # type: ignore[index]
    if not top:
        lines.append("_None flagged._")
    else:
        for item in top:
            kind = str(item["kind"])
            rsk = item["risk_score"]
            cat = item["category"]
            if kind == "changed":
                loc = f"p{item['source_page']}/p{item['target_page']}"
                body = f"`{item['source_snippet']}` -> `{item['target_snippet']}`"
            elif kind == "missing":
                loc = f"p{item['source_page']}"
                body = f"missing: `{item['source_snippet']}`"
            else:
                loc = f"p{item['target_page']}"
                body = f"extra: `{item['target_snippet']}`"
            lines.append(f"- **Risk {rsk}** [{cat}] {loc}: {body}")
    lines.append("")
    lines.append("### Likely extraction noise (review)")
    lines.append("")
    noise = recap["likely_extraction_noise"]  # type: ignore[index]
    if not noise:
        lines.append("_None flagged._")
    else:
        for nb in noise:
            lines.append(
                f"- **Noise {nb['noise_score']}** ({nb['kind']}, {nb['pages']}): {nb['text']}"
            )
    lines.append("")
    ts = recap["top_snippets"]  # type: ignore[index]
    lines.append("### Top snippets (by risk, bounded)")
    lines.append("")
    lines.append("#### Changed")
    lines.append("")
    ch = ts["changed"]  # type: ignore[index]
    if not ch:
        lines.append("_None._")
    else:
        lines.append(
            _md_table(
                ["Risk", "Cat", "Score", "Pg", "Source", "Target"],
                [
                    [
                        str(r["risk_score"]),
                        str(r["category"]),
                        str(r["score"]),
                        f"p{r['source_page']}/p{r['target_page']}",
                        str(r["source_snippet"]),
                        str(r["target_snippet"]),
                    ]
                    for r in ch
                ],
            )
        )
    lines.append("")
    lines.append("#### Missing / extra")
    lines.append("")
    miss = ts["missing"]  # type: ignore[index]
    ext = ts["extra"]  # type: ignore[index]
    miss_rows = [
        [str(r["risk_score"]), str(r["category"]), f"p{r['source_page']}", str(r["snippet"])] for r in miss
    ]
    ext_rows = [
        [str(r["risk_score"]), str(r["category"]), f"p{r['target_page']}", str(r["snippet"])] for r in ext
    ]
    lines.append("Missing:")
    lines.append("")
    lines.append(_md_table(["Risk", "Cat", "Pg", "Snippet"], miss_rows) if miss_rows else "_None._")
    lines.append("")
    lines.append("Extra:")
    lines.append("")
    lines.append(_md_table(["Risk", "Cat", "Pg", "Snippet"], ext_rows) if ext_rows else "_None._")
    lines.append("")
    return lines


def to_markdown(report: dict[str, object], source_name: str, target_name: str) -> str:
    s = report["summary"]  # type: ignore[index]
    changed = report["changed"]  # type: ignore[index]
    missing = report["missing"]  # type: ignore[index]
    extra = report["extra"]  # type: ignore[index]
    ext = report["extraction"]  # type: ignore[index]
    opts = report["options"]  # type: ignore[index]
    recap = report["recap"]  # type: ignore[index]

    lines: list[str] = []
    lines.append("# PDF Text Difference Report")
    lines.append("")
    lines.append(f"Source: `{source_name}`")
    lines.append(f"Target: `{target_name}`")
    lines.append("")
    lines.append("## Run options")
    lines.append("")
    lines.append(
        _md_table(
            ["Option", "Value"],
            [
                ["strictness", str(opts["strictness"])],  # type: ignore[index]
                ["disable_vertical", str(opts["disable_vertical"])],  # type: ignore[index]
                ["match_threshold", str(opts["match_threshold"])],  # type: ignore[index]
                ["min_candidate", str(opts["min_candidate"])],  # type: ignore[index]
                ["ai_summary", str(opts.get("ai_summary", False))],  # type: ignore[index]
                ["ai_model", str(opts.get("ai_model") or "-")],  # type: ignore[index]
            ],
        )
    )
    lines.append("")
    lines.append("## Extraction statistics")
    lines.append("")
    src_e = ext["source"]  # type: ignore[index]
    tgt_e = ext["target"]  # type: ignore[index]
    lines.append("### Source PDF")
    lines.append("")
    lines.append(
        _md_table(
            ["Metric", "Count"],
            [
                ["Raw lines", str(src_e["raw_lines"])],  # type: ignore[index]
                ["Filtered lines (after noise / vertical filter)", str(src_e["filtered_lines"])],  # type: ignore[index]
                ["Noise dropped", str(src_e["noise_dropped"])],  # type: ignore[index]
                ["Vertical skipped (--disable-vertical)", str(src_e["vertical_skipped"])],  # type: ignore[index]
                ["Vertical lines kept", str(src_e["vertical_kept"])],  # type: ignore[index]
                ["Dedupe removed (same page)", str(src_e["dedupe_removed"])],  # type: ignore[index]
                ["Lines compared", str(src_e["lines_after_dedupe"])],  # type: ignore[index]
            ],
        )
    )
    lines.append("")
    lines.append("### Target PDF")
    lines.append("")
    lines.append(
        _md_table(
            ["Metric", "Count"],
            [
                ["Raw lines", str(tgt_e["raw_lines"])],  # type: ignore[index]
                ["Filtered lines (after noise / vertical filter)", str(tgt_e["filtered_lines"])],  # type: ignore[index]
                ["Noise dropped", str(tgt_e["noise_dropped"])],  # type: ignore[index]
                ["Vertical skipped (--disable-vertical)", str(tgt_e["vertical_skipped"])],  # type: ignore[index]
                ["Vertical lines kept", str(tgt_e["vertical_kept"])],  # type: ignore[index]
                ["Dedupe removed (same page)", str(tgt_e["dedupe_removed"])],  # type: ignore[index]
                ["Lines compared", str(tgt_e["lines_after_dedupe"])],  # type: ignore[index]
            ],
        )
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        _md_table(
            ["Metric", "Count"],
            [
                ["Source lines", str(s["source_lines"])],  # type: ignore[index]
                ["Target lines", str(s["target_lines"])],  # type: ignore[index]
                ["Matched", str(s["matched"])],  # type: ignore[index]
                ["Changed", str(s["changed"])],  # type: ignore[index]
                ["Missing on target", str(s["missing"])],  # type: ignore[index]
                ["Extra on target", str(s["extra"])],  # type: ignore[index]
            ],
        )
    )
    lines.append("")
    ai = report.get("ai_summary")
    if isinstance(ai, dict) and ai.get("status") == "ok" and ai.get("bullets"):
        lines.append("## AI summary")
        lines.append("")
        for b in ai["bullets"]:
            lines.append(f"- {b}")
        lines.append("")
    lines.extend(_markdown_recap_section(recap))
    lines.append("## Full detail: changed text")
    lines.append("")
    changed_rows: list[list[str]] = []
    for r in changed[:120]:
        changed_rows.append(
            [
                str(r["score"]),
                str(r["risk_score"]),
                str(r["category"]),
                f"p{r['source_page']}",
                f"p{r['target_page']}",
                _snippet(str(r["source_text"])),
                _snippet(str(r["target_text"])),
            ]
        )
    lines.append(
        _md_table(
            ["Score", "Risk", "Cat", "Src Pg", "Tgt Pg", "Source", "Target"],
            changed_rows,
        )
    )
    lines.append("")
    lines.append("## Full detail: missing on target")
    lines.append("")
    miss_rows = [
        [str(r["risk_score"]), str(r["category"]), f"p{r['source_page']}", _snippet(str(r["source_text"]))]
        for r in missing[:120]
    ]
    lines.append(_md_table(["Risk", "Cat", "Src Pg", "Source"], miss_rows))
    lines.append("")
    lines.append("## Full detail: extra on target")
    lines.append("")
    extra_rows = [
        [str(r["risk_score"]), str(r["category"]), f"p{r['target_page']}", _snippet(str(r["target_text"]))]
        for r in extra[:120]
    ]
    lines.append(_md_table(["Risk", "Cat", "Tgt Pg", "Target"], extra_rows))
    return "\n".join(lines) + "\n"


def to_html(report: dict[str, object], source_name: str, target_name: str) -> str:
    s = report["summary"]  # type: ignore[index]
    changed = report["changed"]  # type: ignore[index]
    missing = report["missing"]  # type: ignore[index]
    extra = report["extra"]  # type: ignore[index]
    ext = report["extraction"]  # type: ignore[index]
    opts = report["options"]  # type: ignore[index]
    recap = report["recap"]  # type: ignore[index]

    def esc(x: object) -> str:
        return html.escape(str(x))

    def rows_changed() -> str:
        out: list[str] = []
        for r in changed[:200]:
            out.append(
                "<tr>"
                f"<td>{esc(r['score'])}</td>"
                f"<td>{esc(r['risk_score'])}</td>"
                f"<td>{esc(r['category'])}</td>"
                f"<td>{esc(r['source_page'])}</td>"
                f"<td>{esc(r['target_page'])}</td>"
                f"<td class='snip'>{esc(_snippet(str(r['source_text'])))}</td>"
                f"<td class='snip'>{esc(_snippet(str(r['target_text'])))}</td>"
                "</tr>"
            )
        return "".join(out) or "<tr><td colspan='7'>None</td></tr>"

    def rows_missing() -> str:
        out: list[str] = []
        for r in missing[:200]:
            out.append(
                "<tr>"
                f"<td>{esc(r['risk_score'])}</td>"
                f"<td>{esc(r['category'])}</td>"
                f"<td>{esc(r['source_page'])}</td>"
                f"<td class='snip'>{esc(_snippet(str(r['source_text'])))}</td>"
                "</tr>"
            )
        return "".join(out) or "<tr><td colspan='4'>None</td></tr>"

    def rows_extra() -> str:
        out: list[str] = []
        for r in extra[:200]:
            out.append(
                "<tr>"
                f"<td>{esc(r['risk_score'])}</td>"
                f"<td>{esc(r['category'])}</td>"
                f"<td>{esc(r['target_page'])}</td>"
                f"<td class='snip'>{esc(_snippet(str(r['target_text'])))}</td>"
                "</tr>"
            )
        return "".join(out) or "<tr><td colspan='4'>None</td></tr>"

    sc = recap["summary_counts"]  # type: ignore[index]
    cbc = sc["changed_by_category"]  # type: ignore[index]
    cat_bits = ", ".join(f"{k}={cbc[k]}" for k in _CATEGORY_ORDER if int(cbc[k]) > 0) or "(none)"

    top_lines: list[str] = []
    for item in recap["top_potential_changes"]:  # type: ignore[index]
        kind = str(item["kind"])
        rsk = item["risk_score"]
        cat = item["category"]
        if kind == "changed":
            loc = f"p{item['source_page']}/p{item['target_page']}"
            body = f"<code>{esc(item['source_snippet'])}</code> -> <code>{esc(item['target_snippet'])}</code>"
        elif kind == "missing":
            loc = f"p{item['source_page']}"
            body = f"missing: <code>{esc(item['source_snippet'])}</code>"
        else:
            loc = f"p{item['target_page']}"
            body = f"extra: <code>{esc(item['target_snippet'])}</code>"
        top_lines.append(
            f"<li><strong>Risk {esc(rsk)}</strong> [{esc(cat)}] {esc(loc)}: {body}</li>"
        )
    top_html = "<ul class='recap'>" + "".join(top_lines) + "</ul>" if top_lines else "<p>None.</p>"

    noise_lines: list[str] = []
    for nb in recap["likely_extraction_noise"]:  # type: ignore[index]
        noise_lines.append(
            f"<li><strong>Noise {esc(nb['noise_score'])}</strong> ({esc(nb['kind'])}, {esc(nb['pages'])}): "
            f"{esc(nb['text'])}</li>"
        )
    noise_html = "<ul class='recap'>" + "".join(noise_lines) + "</ul>" if noise_lines else "<p>None.</p>"

    ts = recap["top_snippets"]  # type: ignore[index]
    top_ch_rows: list[str] = []
    for r in ts["changed"]:  # type: ignore[index]
        top_ch_rows.append(
            "<tr>"
            f"<td>{esc(r['risk_score'])}</td>"
            f"<td>{esc(r['category'])}</td>"
            f"<td>{esc(r['score'])}</td>"
            f"<td>{esc(r['source_page'])}/{esc(r['target_page'])}</td>"
            f"<td class='snip'>{esc(r['source_snippet'])}</td>"
            f"<td class='snip'>{esc(r['target_snippet'])}</td>"
            "</tr>"
        )
    top_ch_html = "".join(top_ch_rows) or "<tr><td colspan='6'>None</td></tr>"

    miss_snip_rows: list[str] = []
    for r in ts["missing"]:  # type: ignore[index]
        miss_snip_rows.append(
            "<tr>"
            f"<td>{esc(r['risk_score'])}</td>"
            f"<td>{esc(r['category'])}</td>"
            f"<td>{esc(r['source_page'])}</td>"
            f"<td class='snip'>{esc(r['snippet'])}</td>"
            "</tr>"
        )
    miss_snip_html = "".join(miss_snip_rows) or "<tr><td colspan='4'>None</td></tr>"

    ext_snip_rows: list[str] = []
    for r in ts["extra"]:  # type: ignore[index]
        ext_snip_rows.append(
            "<tr>"
            f"<td>{esc(r['risk_score'])}</td>"
            f"<td>{esc(r['category'])}</td>"
            f"<td>{esc(r['target_page'])}</td>"
            f"<td class='snip'>{esc(r['snippet'])}</td>"
            "</tr>"
        )
    ext_snip_html = "".join(ext_snip_rows) or "<tr><td colspan='4'>None</td></tr>"

    src_e = ext["source"]  # type: ignore[index]
    tgt_e = ext["target"]  # type: ignore[index]

    ai_block = report.get("ai_summary")
    if isinstance(ai_block, dict) and ai_block.get("status") == "ok" and ai_block.get("bullets"):
        li_parts = "".join(f"<li>{esc(b)}</li>" for b in ai_block["bullets"])
        ai_section_html = f'  <h2>AI summary</h2>\n  <ul class="recap">{li_parts}</ul>\n\n'
    else:
        ai_section_html = ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>PDF Text Diff Report</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .meta {{ margin-bottom: 18px; color: #555; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(140px, 1fr)); gap: 8px; margin: 12px 0 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fafafa; }}
    .k {{ font-size: 12px; color: #666; }}
    .v {{ font-size: 20px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    ul.recap {{ margin: 8px 0 16px 20px; max-width: 960px; }}
    ul.recap li {{ margin-bottom: 8px; }}
    td.snip {{ max-width: 340px; word-break: break-word; font-size: 13px; }}
    .count-inline {{ font-size: 14px; color: #444; margin: 8px 0 16px; }}
  </style>
</head>
<body>
  <h1>PDF Text Difference Report</h1>
  <div class="meta">Source: <code>{esc(source_name)}</code><br/>Target: <code>{esc(target_name)}</code></div>

  <h2>Run options</h2>
  <table>
    <thead><tr><th>Option</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>strictness</td><td>{esc(opts['strictness'])}</td></tr>
      <tr><td>disable_vertical</td><td>{esc(opts['disable_vertical'])}</td></tr>
      <tr><td>match_threshold</td><td>{esc(opts['match_threshold'])}</td></tr>
      <tr><td>min_candidate</td><td>{esc(opts['min_candidate'])}</td></tr>
      <tr><td>ai_summary</td><td>{esc(opts.get('ai_summary', False))}</td></tr>
      <tr><td>ai_model</td><td>{esc(opts.get('ai_model') or '-')}</td></tr>
    </tbody>
  </table>

  <h2>Extraction statistics</h2>
  <div class="grid2">
    <div>
      <h3>Source PDF</h3>
      <table>
        <thead><tr><th>Metric</th><th>Count</th></tr></thead>
        <tbody>
          <tr><td>Raw lines</td><td>{esc(src_e['raw_lines'])}</td></tr>
          <tr><td>Filtered lines</td><td>{esc(src_e['filtered_lines'])}</td></tr>
          <tr><td>Noise dropped</td><td>{esc(src_e['noise_dropped'])}</td></tr>
          <tr><td>Vertical skipped</td><td>{esc(src_e['vertical_skipped'])}</td></tr>
          <tr><td>Vertical lines kept</td><td>{esc(src_e['vertical_kept'])}</td></tr>
          <tr><td>Dedupe removed</td><td>{esc(src_e['dedupe_removed'])}</td></tr>
          <tr><td>Lines compared</td><td>{esc(src_e['lines_after_dedupe'])}</td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <h3>Target PDF</h3>
      <table>
        <thead><tr><th>Metric</th><th>Count</th></tr></thead>
        <tbody>
          <tr><td>Raw lines</td><td>{esc(tgt_e['raw_lines'])}</td></tr>
          <tr><td>Filtered lines</td><td>{esc(tgt_e['filtered_lines'])}</td></tr>
          <tr><td>Noise dropped</td><td>{esc(tgt_e['noise_dropped'])}</td></tr>
          <tr><td>Vertical skipped</td><td>{esc(tgt_e['vertical_skipped'])}</td></tr>
          <tr><td>Vertical lines kept</td><td>{esc(tgt_e['vertical_kept'])}</td></tr>
          <tr><td>Dedupe removed</td><td>{esc(tgt_e['dedupe_removed'])}</td></tr>
          <tr><td>Lines compared</td><td>{esc(tgt_e['lines_after_dedupe'])}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <h2>Summary</h2>
  <div class="cards">
    <div class="card"><div class="k">Source lines</div><div class="v">{esc(s['source_lines'])}</div></div>
    <div class="card"><div class="k">Target lines</div><div class="v">{esc(s['target_lines'])}</div></div>
    <div class="card"><div class="k">Matched</div><div class="v">{esc(s['matched'])}</div></div>
    <div class="card"><div class="k">Changed</div><div class="v">{esc(s['changed'])}</div></div>
    <div class="card"><div class="k">Missing on target</div><div class="v">{esc(s['missing'])}</div></div>
    <div class="card"><div class="k">Extra on target</div><div class="v">{esc(s['extra'])}</div></div>
  </div>

{ai_section_html}  <h2>Recap (potential issues)</h2>
  <h3>Summary counts</h3>
  <table>
    <thead><tr><th>Metric</th><th>Count</th></tr></thead>
    <tbody>
      <tr><td>Changed (below match threshold)</td><td>{esc(sc['changed_total'])}</td></tr>
      <tr><td>Missing on target</td><td>{esc(sc['missing_total'])}</td></tr>
      <tr><td>Extra on target</td><td>{esc(sc['extra_total'])}</td></tr>
      <tr><td>High-risk changed (risk &gt;= 72)</td><td>{esc(sc['high_risk_changed'])}</td></tr>
    </tbody>
  </table>
  <p class="count-inline"><strong>Changed by category:</strong> {esc(cat_bits)}</p>

  <h3>Top potential changes</h3>
  {top_html}

  <h3>Likely extraction noise (review)</h3>
  {noise_html}

  <h3>Top snippets (by risk)</h3>
  <h4>Changed</h4>
  <table>
    <thead><tr><th>Risk</th><th>Cat</th><th>Score</th><th>Pg</th><th>Source</th><th>Target</th></tr></thead>
    <tbody>{top_ch_html}</tbody>
  </table>
  <h4>Missing</h4>
  <table>
    <thead><tr><th>Risk</th><th>Cat</th><th>Pg</th><th>Snippet</th></tr></thead>
    <tbody>{miss_snip_html}</tbody>
  </table>
  <h4>Extra</h4>
  <table>
    <thead><tr><th>Risk</th><th>Cat</th><th>Pg</th><th>Snippet</th></tr></thead>
    <tbody>{ext_snip_html}</tbody>
  </table>

  <h2>Full detail: changed text</h2>
  <table>
    <thead><tr><th>Score</th><th>Risk</th><th>Cat</th><th>Src Pg</th><th>Tgt Pg</th><th>Source</th><th>Target</th></tr></thead>
    <tbody>{rows_changed()}</tbody>
  </table>

  <h2>Full detail: missing on target</h2>
  <table>
    <thead><tr><th>Risk</th><th>Cat</th><th>Src Pg</th><th>Source</th></tr></thead>
    <tbody>{rows_missing()}</tbody>
  </table>

  <h2>Full detail: extra on target</h2>
  <table>
    <thead><tr><th>Risk</th><th>Cat</th><th>Tgt Pg</th><th>Target</th></tr></thead>
    <tbody>{rows_extra()}</tbody>
  </table>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("experiments/manual_runs"))
    ap.add_argument("--run-id", default=None, type=str)
    ap.add_argument(
        "--strictness",
        type=float,
        default=0.92,
        help="0.0-1.0: higher removes more noise and tightens default match thresholds (default: 0.92)",
    )
    ap.add_argument(
        "--disable-vertical",
        action="store_true",
        help="Ignore text lines whose baseline is ~vertical (+/-90 deg from dir).",
    )
    ap.add_argument(
        "--match-threshold",
        type=float,
        default=None,
        help="Fuzz ratio >= this counts as unchanged match (default: derived from --strictness).",
    )
    ap.add_argument(
        "--min-candidate",
        type=float,
        default=None,
        help="Minimum fuzz ratio to pair two lines (default: derived from --strictness).",
    )
    ap.add_argument(
        "--ai-summary",
        action="store_true",
        help="Request a short OpenAI summary (reads OPENAI_API_KEY from the environment).",
    )
    ap.add_argument(
        "--ai-model",
        default="gpt-4o-mini",
        help="Chat Completions model when --ai-summary is set (default: gpt-4o-mini).",
    )
    args = ap.parse_args()

    if not 0.0 <= args.strictness <= 1.0:
        raise SystemExit("--strictness must be between 0.0 and 1.0")

    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source: {source}")
    if not target.is_file():
        raise FileNotFoundError(f"Missing target: {target}")

    match_threshold = (
        float(args.match_threshold) if args.match_threshold is not None else default_match_threshold(args.strictness)
    )
    min_candidate = (
        float(args.min_candidate) if args.min_candidate is not None else default_min_candidate(args.strictness)
    )
    if min_candidate > match_threshold:
        raise SystemExit("--min-candidate must be <= --match-threshold")

    run_id = args.run_id or datetime.now().strftime("textdiff_%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    src_lines, src_stats = extract_pdf_lines(source, strictness=args.strictness, disable_vertical=args.disable_vertical)
    tgt_lines, tgt_stats = extract_pdf_lines(target, strictness=args.strictness, disable_vertical=args.disable_vertical)
    pairs, missing_src, extra_tgt = match_lines(src_lines, tgt_lines, min_candidate=min_candidate)
    run_options: dict[str, object] = {
        "strictness": args.strictness,
        "disable_vertical": args.disable_vertical,
        "match_threshold": match_threshold,
        "min_candidate": min_candidate,
        "ai_summary": bool(args.ai_summary),
        "ai_model": args.ai_model if args.ai_summary else None,
    }
    report = build_report(
        src_lines,
        tgt_lines,
        pairs,
        missing_src,
        extra_tgt,
        match_threshold=match_threshold,
        min_candidate=min_candidate,
        source_stats=src_stats,
        target_stats=tgt_stats,
        run_options=run_options,
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    report["ai_summary"] = try_generate_ai_summary(
        report,
        enabled=bool(args.ai_summary),
        model=str(args.ai_model),
        api_key=api_key,
    )

    json_path = out_dir / "text_diff_report.json"
    md_path = out_dir / "text_diff_report.md"
    html_path = out_dir / "text_diff_report.html"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(report, source.name, target.name), encoding="utf-8")
    html_path.write_text(to_html(report, source.name, target.name), encoding="utf-8")

    print(f"OUT_DIR={out_dir.resolve()}")
    print(f"REPORT_JSON={json_path.resolve()}")
    print(f"REPORT_MD={md_path.resolve()}")
    print(f"REPORT_HTML={html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
