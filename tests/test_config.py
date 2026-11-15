"""
Checks that config/ and prompts/ match the required shapes
(Docs/09 section 3, Docs/12_CONFIGURATION_EXAMPLES.md, Docs/05 section 2).
Each test is skipped until its file exists.
"""

import pytest

from conftest import CATEGORY_CODES, COMBINED_DATASET, CONFIG_DIR, PROMPTS_DIR


SETTINGS = CONFIG_DIR / "settings.yaml"
CATEGORIES = CONFIG_DIR / "major_categories.yaml"


def _load(path):
    yaml = pytest.importorskip("yaml")
    if not path.exists():
        pytest.skip(f"{path.name} does not exist yet")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# config/settings.yaml
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("input_batch_size", 1000),
        ("embedding_batch_size", 32),
        ("sentiment_batch_size", 8),
        ("llm_batch_size", 40),
        ("category_min_similarity", 0.35),
        ("llm_cluster_merge_similarity", 0.82),
        ("random_seed", 42),
        ("max_process_memory_mb", 5120),
    ],
)
def test_settings_top_level_values(key, value):
    settings = _load(SETTINGS)

    assert settings[key] == value


def test_settings_model_ids_are_fixed():
    models = _load(SETTINGS)["models"]

    assert models["embedding"]["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert models["embedding"]["device"] == "cpu"
    assert models["embedding"]["normalize_embeddings"] is True
    assert models["sentiment"]["model_id"] == "cardiffnlp/twitter-roberta-base-sentiment-latest"
    assert models["sentiment"]["device"] == "cpu"
    assert models["llm"]["model_id"] == "gemini-3.6-flash"
    assert models["llm"]["api_key_env_var"] == "GEMINI_API_KEY"
    assert models["llm"]["batch_size"] == 40
    assert 0 < models["llm"]["max_retries"] <= 10


def test_settings_contain_no_api_key():
    text = SETTINGS.read_text(encoding="utf-8") if SETTINGS.exists() else pytest.skip("no settings.yaml")

    assert "AIza" not in text
    assert "api_key:" not in text


def test_priority_weights_and_bands():
    priority = _load(SETTINGS)["priority"]

    assert priority["weights"] == {
        "volume": 0.35, "negative_sentiment": 0.35, "severity": 0.20, "trend": 0.10,
    }
    assert sum(priority["weights"].values()) == pytest.approx(1.0)
    assert priority["priority_bands"] == {"critical": 80, "high": 60, "medium": 35}
    assert priority["trend"]["minimum_reviews_per_window"] == 5


def test_priority_severity_levels():
    severity = _load(SETTINGS)["priority"]["severity"]

    assert {level: severity[level]["score"] for level in ("critical", "high", "medium", "low")} == {
        "critical": 100, "high": 75, "medium": 50, "low": 25,
    }
    for level in severity.values():
        assert level["phrases"], "every severity level needs at least one phrase"


# ---------------------------------------------------------------------------
# config/major_categories.yaml
# ---------------------------------------------------------------------------

def test_taxonomy_matches_default_categories_and_codes():
    categories = _load(CATEGORIES)["categories"]

    assert {c["name"]: c["code"] for c in categories} == CATEGORY_CODES


def test_taxonomy_names_and_codes_are_unique_and_other_exists():
    categories = _load(CATEGORIES)["categories"]
    names = [c["name"] for c in categories]
    codes = [c["code"] for c in categories]

    assert len(names) == len(set(names))
    assert len(codes) == len(set(codes))
    assert "Other" in names


def test_every_category_has_description_and_examples():
    for category in _load(CATEGORIES)["categories"]:
        assert category["description"].strip(), category["name"]
        assert category["examples"], category["name"]


@pytest.mark.skipif(not COMBINED_DATASET.exists(), reason="local dataset not available")
def test_dataset_major_classes_are_in_taxonomy():
    pandas = pytest.importorskip("pandas")
    names = {c["name"] for c in _load(CATEGORIES)["categories"]}

    frame = pandas.read_csv(COMBINED_DATASET, dtype=str, keep_default_na=False)
    gold = {value.strip() for value in frame["Major Class"] if value.strip()}

    assert gold <= names, f"labels missing from taxonomy: {gold - names}"


# ---------------------------------------------------------------------------
# prompts/
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["subcategory_system.txt", "subcategory_user_template.txt"])
def test_prompt_files_exist_and_are_not_empty(name):
    path = PROMPTS_DIR / name
    if not PROMPTS_DIR.exists():
        pytest.skip("prompts/ does not exist yet")

    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()
