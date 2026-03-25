# Drawing Compare

Production-oriented Python service for comparing **general arrangement (GA)** engineering drawings using OCR. It classifies text regions as matched, changed, missing, extra, or uncertain; emits structured comparison payloads and review flags; and supports **evaluation** against a golden dataset.

## Stack

- Python 3.11
- FastAPI, Pydantic, pydantic-settings
- OpenCV (preprocessing)
- RapidFuzz (fuzzy matching)

## Quick start

```bash
cd drawing-compare
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Compare (stub): `POST http://localhost:8000/compare`

## Run a saved compare (local files)

If you want artifact files you can open after a run (JSON + Markdown):

```bash
cd drawing-compare
.venv\Scripts\activate   # Windows
python scripts/run_manual_compare.py ^
  --source data/test_inputs/source.pdf ^
  --target data/test_inputs/target.pdf ^
  --ocr-provider echo
```

Outputs are written under `experiments/manual_runs/<run-id>/`:

- `compare_response.json` (full API payload)
- `comparison_report.json` (reader-friendly report JSON)
- `comparison_report.md` (reader-friendly report markdown)
- `run_summary.json` (small run metadata + counts)

## PDF text-first diff report (embedded text)

For PDFs with embedded text layers, generate a legible changed/missing/extra report:

```bash
cd drawing-compare
.venv\Scripts\activate   # Windows
python scripts/compare_pdf_text.py ^
  --source data/test_inputs/source.pdf ^
  --target data/test_inputs/target.pdf
```

Artifacts (under `experiments/manual_runs/textdiff_<timestamp>/`):

- `text_diff_report.json`
- `text_diff_report.md`
- `text_diff_report.html`

### View the latest text diff report in the browser

1. **Start the API** (from `drawing-compare`):

```bash
.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Generate a text diff** (same shell or another; from `drawing-compare`):

```bash
python scripts/compare_pdf_text.py ^
  --source data/test_inputs/source.pdf ^
  --target data/test_inputs/target.pdf
```

3. **Open the latest HTML report**:

- Browser: `http://localhost:8000/reports/latest-text-diff`
- Or: `curl -s http://localhost:8000/reports/latest-text-diff -o report.html`

The route **`GET /reports/latest-text-diff`** returns the newest `text_diff_report.html` among directories named `experiments/manual_runs/textdiff_*` (by file modification time). If none exists, it returns an HTML page with the same generation command so you can copy-paste it.

## Compare PDFs by embedded text (legible diff now)

When PDFs already contain selectable text, this gives better difference callouts quickly:

```bash
cd drawing-compare
.venv\Scripts\activate   # Windows
python scripts/compare_pdf_text.py ^
  --source data/test_inputs/source.pdf ^
  --target data/test_inputs/target.pdf
```

Outputs:

- `pdf_text_diff.md` (legible report)
- `pdf_text_diff.html` (visual browser report)
- `pdf_text_diff.json` (structured data)

## Project layout

| Path | Role |
|------|------|
| `app/api` | HTTP routers |
| `app/core` | Config, logging, DI-friendly app wiring |
| `app/models` | Pydantic domain schemas |
| `app/preprocessing` | Image normalization, deskew, ROI hints |
| `app/ocr` | Provider abstraction + adapters |
| `app/parsing` | OCR tokens → drawing fields |
| `app/matching` | Pairwise alignment and fuzzy scores |
| `app/classification` | matched / changed / missing / extra / uncertain |
| `app/reporting` | Structured reports and review flags |
| `app/evaluation` | Metrics vs golden pairs |
| `app/datasets` | Manifests and dataset loaders |
| `app/rules` | Tunable business rules for flags |

Data and docs live under `data/`, `docs/`, and `experiments/`.

## Configuration

Environment variables (prefix `DRAWING_COMPARE_`) override defaults. See `app/core/config.py` for thresholds and paths.

## Documentation

- [Labeling guide](docs/labeling_guide.md)
- [Evaluation guide](docs/evaluation_guide.md)
- [Error taxonomy](docs/error_taxonomy.md)

## License

Proprietary — adjust as needed for your organization.
