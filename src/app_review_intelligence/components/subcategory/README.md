# Subcategory Discovery Component (Person 4)

Discovers actionable app issue subcategories from reviews grouped by predicted major category using Gemini, assigning every review ID to exactly one subcategory or `UNASSIGNED`, and consolidating duplicates across batches.

## Usage & Interfaces

```python
from app_review_intelligence.components.subcategory.interface import (
    discover_subcategories,
    discover_subcategories_with_report,
)

records = [{"review_id": "R001", "review_text": "Crashes on login", "predicted_major_category": "Performance and Stability"}]
result, report = discover_subcategories_with_report(records)
# result: {"subcategories": [{"subcategory_id": "PERF_001", "subcategory_name": "App Crashes", "description": "Crash reports.", "major_category": "Performance and Stability"}], "assignments": [{"review_id": "R001", "subcategory_id": "PERF_001"}]}
```

## Report Fields
`report` contains execution counts only (never review text or secrets):
- `batches`, `api_calls`, `retries`: Request tracking metrics.
- `unassigned_review_ids`: List of review IDs marked `UNASSIGNED`.
- `repaired`: List of `{review_id, problem}` entries repaired after exhausted retries.
- `merges`: Count of subcategory groups consolidated across batches.
- `model_id`: Configured LLM identifier.
- `categories`: Mapping of major category to input review count.

## Error & Failure Handling
- **Raises `ValueError`**: Missing/empty API key env var; invalid records structure; empty review ID/text; duplicate review IDs; unknown major category.
- **Raises `RuntimeError`**: Persistent bad JSON, unrecoverable API errors, or major category mismatch after `1 + max_retries` bounded retries.
- **`UNASSIGNED` assignment**: Model explicitly unassigns a review, or semantic defects (missing/duplicate review IDs) are repaired after retries are exhausted.

## Settings & Defaults (`config/settings.yaml`)
- `models.llm.model_id`: `"gemini-3.6-flash"`
- `models.llm.api_key_env_var`: `"GEMINI_API_KEY"` (read via `os.environ` only)
- `models.llm.batch_size`: `40` (max reviews per Gemini call)
- `models.llm.max_retries`: `3` (max retries per batch; total attempts <= 4)
- `llm_cluster_merge_similarity`: `0.82` (MiniLM cosine similarity threshold)
- `models.embedding.model_id`: `"sentence-transformers/all-MiniLM-L6-v2"`
- `random_seed`: `42`
- `models.llm.min_request_interval_seconds`: `0`

## Running Tests
```bash
pytest tests/test_subcategory.py tests/test_subcategory_internals.py -q   # Offline default suite
pytest tests/test_subcategory_internals.py -q -m local_model            # Real MiniLM embedding test
pytest tests/test_subcategory.py -q -m gemini_api                       # Real Gemini API smoke test
```
