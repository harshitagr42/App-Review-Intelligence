# App Review Intelligence

A local command-line tool that reads English app reviews from CSV/XLSX files and, for each review, predicts:

- one **major category** (local MiniLM similarity against `config/major_categories.yaml`),
- a **sentiment** (local CardiffNLP RoBERTa),
- a **subcategory** discovered by Google Gemini inside each predicted major category.

It then ranks subcategories by a transparent **priority heuristic** and writes a Markdown report plus CSV/JSON files into a new output folder for every run.

> Priority scoring (Person 5) is not implemented yet. Runs complete without it and the report says priority is unavailable.

## 1. Requirements

- Python **3.13** (`python3 --version` must print `3.13.x`).
- About 600 MB free disk for the two local models (downloaded once into `models/`).
- A Gemini API key for `run` and `evaluate`.

## 2. Setup

Dependencies are installed with [`uv`](https://docs.astral.sh/uv/) from the committed `uv.lock`.

### macOS (Apple Silicon)

```bash
brew install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv sync                              # creates .venv from uv.lock (includes pytest)
source .venv/bin/activate
python inference.py download-models  # one-time: local models into models/
```

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv sync                              # creates .venv from uv.lock (includes pytest)
source .venv/bin/activate
python inference.py download-models  # one-time: local models into models/
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.13
uv sync
.venv\Scripts\Activate.ps1
python inference.py download-models
```

Windows and Linux steps have not been physically verified on those platforms.

Check the install:

```bash
python inference.py --help
```

## 3. Gemini API key

The key is read from the `GEMINI_API_KEY` environment variable. It is never accepted as a command-line argument, printed or written to any output.

Easiest: copy the template and paste your key into `.env` (git-ignored):

```bash
cp .env.example .env        # then edit .env: GEMINI_API_KEY=...
```

Or set it in the shell:

```bash
export GEMINI_API_KEY="your-key"          # macOS / Linux
```

```powershell
$env:GEMINI_API_KEY = "your-key"          # Windows PowerShell
```

A variable already set in the shell wins over `.env`.

**Privacy warning:** `run` and `evaluate` send review IDs, review text and the predicted major category to the Gemini API. On the free tier, Google may use submitted content to improve its products. Do not process confidential reviews without permission. Both commands refuse to start without `--allow-cloud-processing`.

## 4. Models

| Role | Model | Runs | License |
|---|---|---|---|
| Embeddings / similarity | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | locally, CPU | Apache-2.0 |
| Sentiment | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | locally, CPU | CC-BY-4.0 (Cardiff NLP) |
| Subcategory discovery | `gemini-3.6-flash` | Google Gemini API | — |

The two local models live in the project's `models/` folder (git-ignored, about 570 MB) and are always loaded from there, so runs work offline. `python inference.py download-models` fetches them once (`--force` re-downloads); if they are missing, the first run downloads them automatically. The downloaded commit hash is saved in `models/<model>/REVISION` and recorded in `run_metadata.json`. Set `APP_REVIEW_INTELLIGENCE_MODELS_DIR` to keep them elsewhere. They are never loaded at the same time. No model is trained or fine-tuned. Model IDs are fixed in `config/settings.yaml`, and the program refuses a different ID.

## 5. Commands

Every command supports `--help`. Shared input options:

| Option | Default | Meaning |
|---|---|---|
| `--input PATH` | required | CSV or XLSX file; repeat to combine several files |
| `--text-column NAME` | `Review` | review-text column (the only required column) |
| `--id-column NAME` | `ID` | review-ID column (IDs are generated if missing) |
| `--date-column NAME` | `Date` | review-date column |
| `--sheet NAME` | first sheet | worksheet for XLSX files |

### validate: check files, no models, no Gemini, no output folder

```bash
python inference.py validate --input dataset/processed/combined_reviews_output.csv
python inference.py validate --input reviews_a.csv --input reviews_b.xlsx --sheet Reviews
python inference.py validate --input export.csv --text-column Comment --id-column ReviewId --date-column Posted
```

### run: full pipeline

```bash
python inference.py run --input reviews_a.csv --input reviews_b.xlsx --allow-cloud-processing
```

Extra options: `--config-dir PATH` (default `config/`), `--output-dir PATH` (default `outputs/`), `--quiet`, `--verbose`. Gold-label columns in the input are ignored by `run`.

### evaluate: run the pipeline on the labelled file and score it

```bash
python inference.py evaluate --input dataset/processed/combined_reviews_output.csv --allow-cloud-processing
```

Extra options: `--seed 42`, `--config-dir`, `--output-dir`, `--quiet`, `--verbose`. The input must contain `Major Class`, `Minor Class` and `Sentiment`.

To re-score an existing run without calling any model or Gemini:

```bash
python inference.py evaluate --run-dir outputs/run_20261115T101500Z_a1b2c3d4 --input dataset/processed/combined_reviews_output.csv
```

### download-models: fetch the local models into models/

```bash
python inference.py download-models
```

Before `run`/`evaluate` touch any file, they make one metadata-only request to Gemini (no review text) to check the key, network and model. If it fails they stop with exit code 5 and create no output folder. `--config-dir` is used by the whole pipeline (settings, taxonomy and Gemini settings).

### Exit codes

`0` success · `1` general failure · `2` invalid arguments/configuration (incl. missing consent or API key) · `3` invalid input file · `4` local model failure · `5` Gemini API/network/quota failure · `6` evaluation failure · `7` memory limit reached.

## 6. Outputs

Every run creates a new folder. Existing folders are never overwritten.

```text
outputs/run_<YYYYMMDDTHHMMSSZ>_<8 chars>/
    report.md               ten-section report (summary, distributions, priority, subcategories, methodology, limitations)
    classified_reviews.csv  one row per accepted review with every prediction (no gold labels)
    clusters.json           subcategories grouped by major category, members, sentiment counts, unassigned IDs
    run_metadata.json       inputs + SHA-256, row counts, model IDs/revisions, seed, stage times, peak RSS, API counts, errors
    logs.txt                timestamped logs without API keys or review text

outputs/eval_<YYYYMMDDTHHMMSSZ>_<8 chars>/
    evaluation_report.md    metrics, confusion matrices, exclusions, runtime, limitations
    evaluation.json         raw metrics
    run_metadata.json
    logs.txt
```

A failed run keeps its folder with `logs.txt` and `run_metadata.json` (`"status": "failed"`). Use `--verbose` to add the traceback to `logs.txt`.

Rows with an empty review, an empty/invalid date, or the same text and date as an earlier row are removed in memory (the input file is never changed) and counted. For the bundled dataset: 2,500 rows read and 1,997 accepted.

## 7. Common errors

| Message | Fix |
|---|---|
| `Re-run with --allow-cloud-processing` | Add the flag to consent to sending text to Gemini. |
| `GEMINI_API_KEY is not set` | Put the key in `.env` or export it (section 3). |
| `must contain a 'Review' column` | Use `--text-column` to name your text column. |
| `unsupported file format` | Use `.csv` or `.xlsx`. |
| `Stage 'api_check' failed` / `Stage 'subcategory' failed` (exit 5) | Check the key, network and Gemini quota, then run again. |
| `Stage 'sentiment' / 'major_category' failed` (exit 4) | The model could not be loaded; run `python inference.py download-models`. |

## 8. Limits

- The memory target is 5 GB (`max_process_memory_mb`); the run stops with exit code 7 if peak RSS exceeds it. It has not yet been measured on the full dataset with real Gemini calls.
- Runtime depends on Gemini rate limits and network latency, which cannot be guaranteed.
- Subcategories come from an LLM and can differ between runs.
- Priority is a heuristic, not a validated measure.

## 9. Development

```bash
source .venv/bin/activate
pytest                                     # full suite; Gemini is always mocked
pytest -m "not local_model"                # skip tests that load the local models
```

Each component in `src/app_review_intelligence/` has an `implementation.py` (logic) and an `interface.py` (plain functions). Components import each other only through `interface.py`. The pipeline order is: input adapter → preprocessing → major category → sentiment → subcategory → priority → evaluation (evaluate) / reporting (run).
