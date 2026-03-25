"""Reusable regex-backed :class:`LineFieldExtractor` implementation."""

from __future__ import annotations

import re

from app.parsing.line_fields.base import LineFieldExtractor
from app.parsing.line_fields.context import LineExtractionContext
from app.parsing.line_fields.matches import ExtractionMatch


class RegexLineExtractor(LineFieldExtractor):
    """Single pattern with named ``label``/``value`` groups or numeric ``(label, value)`` groups."""

    def __init__(
        self,
        rule_id: str,
        pattern: str,
        field_type: str,
        rule_weight: float = 0.9,
        *,
        flags: int = re.IGNORECASE | re.UNICODE,
        match_mode: str = "full",
        implicit_label: str | None = None,
    ) -> None:
        self._rule_id = rule_id
        self._pattern = re.compile(pattern, flags)
        self._field_type = field_type
        self._weight = rule_weight
        self._mode = match_mode  # "full" -> match, "findall" -> finditer
        self._implicit_label = implicit_label

    @property
    def rule_id(self) -> str:
        return self._rule_id

    def extract(self, ctx: LineExtractionContext) -> list[ExtractionMatch]:
        text = ctx.text
        if not text:
            return []

        if self._mode == "full":
            m = self._pattern.match(text)
            if not m:
                return []
            return [self._to_match(m, text)]

        out: list[ExtractionMatch] = []
        for m in self._pattern.finditer(text):
            out.append(self._to_match(m, text))
        return out

    def _to_match(self, m: re.Match[str], full_text: str) -> ExtractionMatch:
        if self._implicit_label is not None:
            label = self._implicit_label
            gd = m.groupdict()
            if "value" in gd and gd["value"] is not None:
                value = str(gd["value"]).strip()
            elif m.lastindex and m.lastindex >= 1:
                value = str(m.group(1)).strip()
            else:
                raise ValueError("value-only pattern missing value group")
        else:
            label, value = _label_value_from_match(m)
        raw = m.group(0).strip() or full_text
        return ExtractionMatch(
            field_type=self._field_type,
            label=label,
            value=value,
            raw_text=raw,
            rule_id=self._rule_id,
            rule_weight=self._weight,
        )


def _label_value_from_match(m: re.Match[str]) -> tuple[str, str]:
    gd = m.groupdict()
    if "label" in gd and "value" in gd and gd["label"] is not None and gd["value"] is not None:
        return str(gd["label"]).strip(), str(gd["value"]).strip()
    if "l" in gd and "v" in gd and gd["l"] is not None and gd["v"] is not None:
        return str(gd["l"]).strip(), str(gd["v"]).strip()
    if m.lastindex and m.lastindex >= 2:
        g1 = m.group(1)
        g2 = m.group(2)
        if g1 is not None and g2 is not None:
            return str(g1).strip(), str(g2).strip()
    raise ValueError("Regex match missing label/value groups")
