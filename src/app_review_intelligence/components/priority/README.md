# Priority Component (Person 5)

Scores how urgent each discovered subcategory is, using review counts, predicted sentiment, severity phrases and (optionally) review dates.

## Usage

```python
from app_review_intelligence.components.priority.interface import calculate_priorities

results = calculate_priorities(subcategory_result, sentiment_results)                   # minimum
results = calculate_priorities(subcategory_result, sentiment_results, model_records)    # adds review-text severity + trend
```

- `subcategory_result`: `{"subcategories": [...], "assignments": [...]}` (writing_block.md 5E).
- `sentiment_results`: `[{"review_id", "sentiment", "confidence"}]`.
- `review_records` (optional): model-facing records `{"review_id", "review_text", "review_date"}`. Never pass gold labels.
- `config` (optional): `{"priority": {...}}` overrides `config/settings.yaml`; `{"config_dir": path}` picks another folder.

Returns one dict per subcategory, highest score first: `subcategory_id`, `subcategory_name`, `review_count`, `negative_proportion`, `priority_score`, `priority_level`, `explanation`, `components`.

## How the score is produced

`score = 0.35*V + 0.35*N + 0.20*S + 0.10*T` (weights in `config/settings.yaml`), clamped to 0-100 and rounded to 2 decimals.

| Part | Meaning | Calculation |
|---|---|---|
| V volume | How many reviews, relative to its own major category | `100 * ln(1+n) / ln(1+n_max)`, `n_max` = largest subcategory in the same major category |
| N negative | How negative the group is | `100 * negative reviews / reviews that have a sentiment` |
| S severity | How serious the wording is | 25 / 50 / 75 / 100. Phrase lists in config; highest matching level wins; no match gives 25 |
| T trend | Are complaints growing? | Negative reviews per day, last 30 days vs the 30 days before. 50 = flat, 100 = rising fast, 0 = falling fast |

Levels: score >= 80 Critical, >= 60 High, >= 35 Medium, otherwise Low (bands in config).

## Rules and edge cases

- `UNASSIGNED` reviews are ignored. A positive group with high volume can never reach Critical because N is 0.
- Severity text: the subcategory name and description are always searched. If `review_records` is given, the group's review texts are searched too. Matching ignores case and punctuation and treats "can't", "cant", "login" and "authorization" like the config spellings. `components.severity_reason_codes` lists the matches (for example `high:cannot log in`) and `severity_source` says what was searched.
- Trend needs `review_records` with dates. The reference date is the newest review date in the data (not today). A window needs at least 5 reviews (config) in the subcategory, otherwise trend is `None`, `trend_status` says why, and the trend weight is shared across V, N and S so weights still sum to 1 (`components.effective_weights`).
- Assigned reviews without a sentiment result are left out of the negative share (a warning is logged).
- Bad input raises `TypeError` or `ValueError`: duplicate review IDs, assignments to unknown subcategories, unknown sentiment labels, invalid settings.
- Output is deterministic and the inputs are never modified. This is a transparent heuristic, not a validated score; there are no priority labels to evaluate it against.

## Tests

```bash
pytest tests/test_priority.py tests/test_priority_extras.py -q
```
