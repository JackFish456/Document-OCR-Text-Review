"""Concrete regex rules for engineering drawing OCR lines."""

from __future__ import annotations

import re

from app.parsing.line_fields.base import LineFieldExtractor
from app.parsing.line_fields.context import LineExtractionContext
from app.parsing.line_fields.matches import ExtractionMatch
from app.parsing.line_fields.regex_extractor import RegexLineExtractor


def default_line_extractors() -> list[LineFieldExtractor]:
    """Ordered registry: specific rules first, generic label-value last."""
    return [
        # --- Title block (high confidence) ---
        RegexLineExtractor(
            "tb_drawn_checked",
            r"^\s*(?P<label>DRAWN\s*BY|CHECKED\s*BY|REVIEWED\s*BY|DESIGNED\s*BY|APPROVED\s*BY)"
            r"\s*[:\-]\s*(?P<value>.+?)\s*$",
            "title_block.signature",
            0.94,
        ),
        RegexLineExtractor(
            "tb_project_job",
            r"^\s*(?P<label>PROJECT|JOB(\s*NO\.?)?|CONTRACT(\s*NO\.?)?|CLIENT|OWNER|ENGINEER)"
            r"\s*[:\-]\s*(?P<value>.+?)\s*$",
            "title_block.project",
            0.93,
        ),
        RegexLineExtractor(
            "tb_date_sheet",
            r"^\s*(?P<label>DATE|SHEET(\s*(NO\.?|NUMBER|#))?|SHEETS?)"
            r"\s*[:\-]\s*(?P<value>.+?)\s*$",
            "title_block.sheet_meta",
            0.93,
        ),
        RegexLineExtractor(
            "tb_scale",
            r"^\s*(?P<label>SCALE)\s*[:\-]\s*(?P<value>.+?)\s*$",
            "title_block.scale",
            0.95,
        ),
        RegexLineExtractor(
            "tb_title",
            r"^\s*(?P<label>DRAWING\s*TITLE|TITLE)\s*[:\-]\s*(?P<value>.+?)\s*$",
            "title_block.title",
            0.92,
        ),
        # --- Drawing identifiers ---
        RegexLineExtractor(
            "dwg_number",
            r"^\s*(?P<label>DWG|DRAWING)(\s*(NO\.?|NUMBER|#))?\s*[:\-]\s*(?P<value>.+?)\s*$",
            "drawing_id.dwg_no",
            0.94,
        ),
        RegexLineExtractor(
            "sheet_id",
            r"^\s*(?P<label>SHEET)\s*(ID|NO\.?|NUMBER|#)?\s*[:\-]\s*(?P<value>.+?)\s*$",
            "drawing_id.sheet",
            0.93,
        ),
        RegexLineExtractor(
            "plan_id",
            r"^\s*(?P<label>PLAN|KEY\s*PLAN)\s*[:\-]\s*(?P<value>.+?)\s*$",
            "drawing_id.plan",
            0.88,
        ),
        # --- Revision ---
        RevisionLineExtractor(),
        # --- Notes ---
        RegexLineExtractor(
            "note_header",
            r"^\s*(?P<label>(?:GENERAL\s+)?NOTES?|NOTES?\s*CONT(?:\.)?D)\s*[:\-]?\s*(?P<value>.*)\s*$",
            "note.header",
            0.9,
        ),
        RegexLineExtractor(
            "note_numbered",
            r"^\s*(?P<label>\d{1,2}[.)])\s+(?P<value>.{3,})\s*$",
            "note.bullet",
            0.82,
        ),
        # --- Elevations / dimensions (line-shaped) ---
        RegexLineExtractor(
            "elev_line",
            r"^\s*(?P<label>EL\.?|ELEV\.?|ELEVATION)\s*[:\-]?\s*(?P<value>[+\-]?[\d.,'\u2032\u2033\u201d\u201c\s\"′″/-]+)\s*$",
            "dimension.elevation",
            0.9,
        ),
        RegexLineExtractor(
            "dim_imperial_run",
            r"^\s*(?P<value>\d{1,3}\s*['′\u2032]?\s*[-–—]\s*\d{1,3}\s*[\"″\u2033'])\s*$",
            "dimension.imperial",
            0.85,
            implicit_label="",
        ),
        # --- Generic label : value (lower weight, last) ---
        RegexLineExtractor(
            "generic_kv",
            r"^\s*(?P<label>[A-Z][A-Z0-9 /&'.]{1,40}?)\s*[:\-]\s+(?P<value>.{2,})\s*$",
            "generic.label_value",
            0.72,
        ),
    ]


class RevisionLineExtractor(LineFieldExtractor):
    """Standalone revision tokens: REV B, R3, REVISION 02."""

    @property
    def rule_id(self) -> str:
        return "revision_standalone"

    def extract(self, ctx: LineExtractionContext) -> list[ExtractionMatch]:
        text = ctx.text
        if not text:
            return []
        m = re.match(
            r"^\s*(?:REV(?:ISION)?|R\.?)\s*[.:\#\-]?\s*(?P<value>[A-Za-z0-9]+(?:\.[0-9]+)?)\s*$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return []
        val = m.group("value").strip()
        return [
            ExtractionMatch(
                field_type="revision",
                label="REV",
                value=val,
                raw_text=m.group(0).strip(),
                rule_id=self.rule_id,
                rule_weight=0.91,
            )
        ]
