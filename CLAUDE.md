# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

App Review Intelligence is a local Python 3.13 CLI. It reads English app reviews from one or more CSV/XLSX files, and for each review it predicts exactly one major category, a sentiment, and a subcategory that Gemini discovers. It then scores priority per subcategory and writes a Markdown report plus CSV/JSON outputs into a new, unique run folder.

**Implementation is in progress.** Person 1's components (`adapters/input/`, `preprocessing/`) exist; the other components are not written yet, and their tests are skipped. The specification lives in `Docs/`. Start with `Docs/00_READ_ME_FIRST.md`, then read `Docs/writing_block.md`.

**`Docs/writing_block.md` is the source of truth.** If another doc in `Docs/` contradicts it, `writing_block.md` wins, and the other doc should be corrected to match.

**Never commit, push or upload `Docs/` or `dataset/`, under any circumstances.** `Docs/` is excluded through `.git/info/exclude` and `dataset/` through `.gitignore`. Before any commit, check that nothing staged is under either folder. Never commit `.env` or API keys either.

## Commands

```bash
source .venv/bin/activate                     # local venv (Python 3.13)
pytest                                        # run all tests (pythonpath=src via pyproject)
pytest tests/test_sentiment.py                # one file
pytest tests/test_pipeline.py::test_pipeline_rejects_missing_input   # one test
```

CI (`.github/workflows/test.yml`) runs `pip install .`, then `pip install pandas pytest`, then `pytest` on Python 3.13. `pyproject.toml` currently holds only pytest settings, so add dependencies there when you introduce them. The planned package manager is `uv`, with a committed `uv.lock`. No linter is configured.

## Architecture

The package is `src/app_review_intelligence/`. Each component is a folder with two files:

- `implementation.py` holds the real logic.
- `interface.py` only imports from `implementation.py` and exposes plain functions. Put no logic in it.

Other components import **only from `interface.py`**. Do not use classes, inheritance, decorators or abstract methods for component interfaces.

Each test calls a public function through `tests/conftest.py::get_component_function`. That helper **skips** the test when the module or function doesn't exist yet. If the module exists but raises an import error for any other reason, the test fails. So a component's tests start running as soon as its interface function is defined. The required functions are:

| Module (under `app_review_intelligence.`) | Function |
|---|---|
| `adapters.input.interface` | `load_and_prepare(input_paths)` → model_records, evaluation_labels, dataset_metadata |
| `preprocessing.interface` | `preprocess_records(records)` |
| `components.major_category.interface` | `classify(records)` |
| `components.sentiment.interface` | `analyze(records)` |
| `components.subcategory.interface` | `discover_subcategories(records_with_major_categories)` |
| `components.priority.interface` | `calculate_priorities(subcategories, sentiment_results)` |
| `evaluation.interface` | `evaluate_predictions(...)` |
| `reporting.interface` | `generate_report(...)` |
| `pipeline.interface` | `run_pipeline(input_paths, config)`: must raise `ValueError`/`FileNotFoundError` on empty input |

Check the matching `tests/test_*.py` file for the exact argument and return shapes before you implement a component.

The pipeline runs in this order:

1. Input adapter: combines the files and splits gold labels off.
2. Preprocessing.
3. Major category: MiniLM cosine similarity against `config/major_categories.yaml`, with `Other` as the fallback.
4. Sentiment: CardiffNLP, run locally and independently of the category steps.
5. Subcategory: reviews are grouped by predicted major category and sent to Gemini in batches of 40, then batches are merged with MiniLM similarity of at least 0.82.
6. Priority: the V/N/S/T formula.
7. Evaluation.
8. Reporting.

`pipeline/` only connects components through their interfaces. `inference.py` at the repo root is the Typer CLI, with the commands `validate`, `run` and `evaluate`.

Configuration lives in two files: `config/settings.yaml` (thresholds, batch sizes, `models` and `priority` sections) and `config/major_categories.yaml`. The ten category names match the dataset's `Major Class` labels exactly. Gemini prompts live in `prompts/*.txt`.

Shared record shapes are plain dicts. Use these field names everywhere (full spec: `writing_block.md` §5, `Docs/05_DATA_SCHEMA.md`):
- Model-facing record: `review_id`, `review_text`, `review_date`
- Evaluation labels: `review_id`, `gold_major_category`, `gold_minor_category`, `gold_sentiment`
- Major category: `review_id`, `predicted_major_category`, `confidence`
- Sentiment: `review_id`, `sentiment` (`Positive`/`Negative`/`Neutral`), `confidence`
- Subcategory result: `{"subcategories": [{subcategory_id, subcategory_name, description, major_category}], "assignments": [{review_id, subcategory_id}]}`. IDs look like `PERF_001`; a review that can't be grouped gets `UNASSIGNED`.
- Priority: `subcategory_id`, `subcategory_name`, `review_count`, `negative_proportion`, `priority_score`, `priority_level`, `explanation`, plus a `components` object

## Hard rules

- **Label isolation:** gold labels (`Major Class`, `Minor Class`, `Sentiment`) are separated out at ingestion. Only `evaluation/` may read them. They must never reach a prediction component or Gemini. `review_id` is used only as an identifier, never as a feature.
- **Input columns:** the defaults are `ID`, `Review` and `Date`, and each can be mapped to another name with a flag. There are no rating or app-name fields.
- **Gemini:** read the key from `GEMINI_API_KEY` and never log or export it. Require the `--allow-cloud-processing` flag before sending anything. Send only the review ID, the review text and the predicted major category. Treat responses as untrusted: every input ID must appear exactly once or be marked `UNASSIGNED`, and no review may be silently dropped. Use bounded retries. Gemini is the only subcategory engine; there is no local clustering engine.
- **Models (fixed, do not substitute):** embeddings use `sentence-transformers/all-MiniLM-L6-v2`; sentiment uses `cardiffnlp/twitter-roberta-base-sentiment-latest`; the LLM is `gemini-3.6-flash`. No training or fine-tuning. Never keep both local models loaded at the same time.
- **Outputs:** each run gets a new folder (`outputs/run_<UTC>_<suffix>/` containing `report.md`, `classified_reviews.csv`, `clusters.json`, `run_metadata.json`, `logs.txt`). Never overwrite an existing folder. Use random seed 42.
- **Dataset:** the only dataset is `dataset/processed/combined_reviews_output.csv` (2,500 rows), for development, demos and evaluation. `dataset/raw/*.csv` were only used to build it and are not used. The file deliberately contains invalid rows (empty review and/or date) and duplicates (same text + same date) to test preprocessing, which removes them in memory (first duplicate kept) and never modifies the file: 2,500 read → 1,997 accepted.
- **Evaluation:** one result set over the accepted reviews, with gold labels from the same file. There is no original/variant split. Subcategories are scored with clustering metrics (ARI/AMI/NMI/V-measure), never called "accuracy". Priority has no ground truth, so report it descriptively only. Never invent metrics.
- **Scope:** do not add a GUI, web framework, database, vector DB, LangChain/LlamaIndex or Docker.
