"""Deterministic tests for scripts/compare_pdf_text.py (synthetic PDF fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import fitz

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import compare_pdf_text as cpt  # noqa: E402


def _write_pdf(path: Path, segments: list[tuple[str, int, tuple[float, float]]]) -> None:
    """Write one page; each segment is (text, rotate_deg, (x, y))."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    for text, rotate, xy in segments:
        page.insert_text(xy, text, fontsize=12, rotate=rotate)
    doc.save(path)
    doc.close()


def test_vertical_text_is_included_when_enabled(tmp_path: Path) -> None:
    pdf = tmp_path / "vert.pdf"
    _write_pdf(
        pdf,
        [
            ("HORIZ", 0, (50.0, 100.0)),
            ("VERT", 90, (200.0, 50.0)),
        ],
    )

    lines, stats = cpt.extract_pdf_lines(pdf, strictness=0.92, disable_vertical=False)
    assert stats.vertical_kept >= 1
    assert any(tl.is_vertical for tl in lines)
    vert_texts = {tl.text for tl in lines if tl.is_vertical}
    assert any("VERT" in t for t in vert_texts)

    lines_skip, stats_skip = cpt.extract_pdf_lines(pdf, strictness=0.92, disable_vertical=True)
    assert stats_skip.vertical_skipped >= 1
    assert not any(tl.is_vertical for tl in lines_skip)
    assert not any("VERT" in tl.text for tl in lines_skip)


def test_noise_filter_removes_garbage_not_short_meaningful_tokens(tmp_path: Path) -> None:
    pdf = tmp_path / "noise.pdf"
    garbage = "\ufffd" * 8
    pipes = "|" * 14
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    y = 40.0
    for label in ("E", '6"', "0' - 6\"", garbage, pipes, "normal label"):
        page.insert_text((40.0, y), label, fontsize=11)
        y += 22.0
    doc.save(pdf)
    doc.close()

    lines, stats = cpt.extract_pdf_lines(pdf, strictness=0.92, disable_vertical=False)
    texts = {tl.text.strip() for tl in lines}

    assert "E" in texts
    assert any(t.startswith("6") for t in texts if '"' in t or t.startswith("6"))
    assert any("0'" in t and "6" in t for t in texts)
    assert "normal label" in texts
    assert not any("\ufffd" in t for t in texts)
    assert stats.noise_dropped >= 2


def test_small_numeric_changes_are_flagged(tmp_path: Path) -> None:
    """Month-name dates avoid pure-numeric slash strings that classify as dimension first."""
    src = tmp_path / "src_date.pdf"
    tgt = tmp_path / "tgt_date.pdf"
    _write_pdf(src, [("Jan 15, 2024 note", 0, (50.0, 80.0))])
    _write_pdf(tgt, [("Feb 16, 2024 note", 0, (50.0, 80.0))])

    s_lines, s_stats = cpt.extract_pdf_lines(src, strictness=0.92, disable_vertical=False)
    t_lines, t_stats = cpt.extract_pdf_lines(tgt, strictness=0.92, disable_vertical=False)
    strictness = 0.92
    match_threshold = cpt.default_match_threshold(strictness)
    min_candidate = cpt.default_min_candidate(strictness)
    pairs, miss, extra = cpt.match_lines(s_lines, t_lines, min_candidate=min_candidate)
    report = cpt.build_report(
        s_lines,
        t_lines,
        pairs,
        miss,
        extra,
        match_threshold=match_threshold,
        min_candidate=min_candidate,
        source_stats=s_stats,
        target_stats=t_stats,
        run_options={
            "strictness": strictness,
            "disable_vertical": False,
            "match_threshold": match_threshold,
            "min_candidate": min_candidate,
            "ai_summary": False,
            "ai_model": None,
        },
    )

    assert report["summary"]["changed"] == 1
    assert float(report["changed"][0]["score"]) < match_threshold
    assert report["changed"][0]["category"] == "date"


def test_recap_is_concise_and_prioritizes_high_risk() -> None:
    changed: list[dict[str, object]] = []
    for i in range(12):
        risk = 50 + i * 3
        changed.append(
            {
                "score": 88.0,
                "source_page": 1,
                "target_page": 1,
                "source_text": f"line{i}a",
                "target_text": f"line{i}b",
                "source_vertical": False,
                "target_vertical": False,
                "category": "generic_text",
                "risk_score": risk,
            }
        )
    recap = cpt.build_recap_payload(
        changed,
        [],
        [],
        match_threshold=92.0,
        min_candidate=70.0,
    )

    top = recap["top_potential_changes"]
    assert len(top) <= cpt._RECAP_TOP_CHANGES
    assert len(top) == 8
    max_risk = max(int(r["risk_score"]) for r in changed)
    assert int(top[0]["risk_score"]) == max_risk
    risks = [int(item["risk_score"]) for item in top]
    assert risks == sorted(risks, reverse=True)

    noise = recap["likely_extraction_noise"]
    assert len(noise) <= cpt._RECAP_NOISE_MAX


def test_ai_grounding_rejects_invalid_numbers() -> None:
    report: dict[str, object] = {
        "summary": {
            "source_lines": 2,
            "target_lines": 2,
            "matched": 1,
            "changed": 1,
            "missing": 0,
            "extra": 0,
        },
        "options": {
            "strictness": 0.92,
            "match_threshold": 91.96,
            "min_candidate": 70.24,
            "ai_summary": True,
            "ai_model": "gpt-4o-mini",
        },
        "recap": {
            "summary_counts": {
                "changed_total": 1,
                "missing_total": 0,
                "extra_total": 0,
                "changed_by_category": {
                    "dimension": 0,
                    "date": 1,
                    "revision_status": 0,
                    "identifier_code": 0,
                    "generic_text": 0,
                },
                "high_risk_changed": 1,
            },
            "top_potential_changes": [
                {
                    "kind": "changed",
                    "risk_score": 88,
                    "category": "date",
                    "fuzz_score": 80.0,
                    "source_page": 1,
                    "target_page": 1,
                    "source_snippet": "01/15/2024",
                    "target_snippet": "02/16/2024",
                }
            ],
        },
    }

    bad_reply = (
        "- The diff shows 999999 separate drawing revisions.\n"
        "- Do not trust this output.\n"
    )
    with patch.object(cpt, "openai_chat_completion", return_value=bad_reply):
        out = cpt.try_generate_ai_summary(
            report,
            enabled=True,
            model="gpt-4o-mini",
            api_key="sk-test-dummy",
        )
    assert out["status"] == "grounding_failed"
    assert out["bullets"] is None
    assert "999999" in (out.get("grounding_violations") or [])

    good_reply = (
        "- Date strings differ between paired lines as in JSON snippets.\n"
        "- One changed pair is categorized as date with risk 88.\n"
    )
    with patch.object(cpt, "openai_chat_completion", return_value=good_reply):
        out_ok = cpt.try_generate_ai_summary(
            report,
            enabled=True,
            model="gpt-4o-mini",
            api_key="sk-test-dummy",
        )
    assert out_ok["status"] == "ok"
    assert out_ok["bullets"] is not None
    assert len(out_ok["bullets"]) <= 8

    no_key = cpt.try_generate_ai_summary(report, enabled=True, model="gpt-4o-mini", api_key=None)
    assert no_key["status"] == "skipped_no_key"
