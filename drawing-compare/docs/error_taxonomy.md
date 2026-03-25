# Error taxonomy

Use consistent categories when reviewing incorrect comparisons or flagging dataset gaps.

## OCR errors

- **OCR_SUBSTITUTION** — Wrong characters due to font similarity (e.g. `0` vs `O`).
- **OCR_MERGE** — Two adjacent strings fused into one region.
- **OCR_SPLIT** — One logical string split across regions.

## Alignment errors

- **FIELD_MISMATCH** — Wrong A/B pairing (greedy alignment picked a suboptimal neighbor).
- **SPATIAL_DRIFT** — Text moved on the sheet; spatial constraint could not reconcile.

## Parsing errors

- **TITLE_BLOCK_MAP** — Incorrect mapping from OCR cluster to semantic field (e.g. date vs scale).
- **NOISE_AS_FIELD** — Non-semantic graphics text treated as a field.

## Drawing / process issues

- **SCAN_ARTIFACT** — Skew, fold marks, or compression blocking OCR.
- **OUT_OF_ROI** — Text clipped by crop or sheet boundary.

## Product / rules

- **THRESHOLD_TUNING** — Correct text pair misclassified due to fuzzy cutoff.
- **MISSING_RULE** — Legitimate review pattern not yet encoded in `ReviewRulesEngine`.

## Severity mapping

- **ERROR** — Likely wrong business decision if trusted automatically (e.g. very low similarity).
- **WARNING** — Needs review (`uncertain`, missing/extra fields).
- **INFO** — Contextual only (extras that may be benign).
