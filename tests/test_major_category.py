"""
Tests for components/major_category (Person 2).

The classifier uses MiniLM cosine similarity against config/major_categories.yaml
(Docs/07_SUBCATEGORY_AND_PRIORITY_RULES.md, section 1).
"""

import pytest

from conftest import CONFIG_DIR, GOLD_FIELDS, MAJOR_CATEGORIES, get_component_function


pytestmark = pytest.mark.local_model

CLASSIFY = (
    "app_review_intelligence.components.major_category.interface",
    "classify",
)


def _category_min_similarity():
    yaml = pytest.importorskip("yaml")
    path = CONFIG_DIR / "major_categories.yaml"
    if not path.exists():
        pytest.skip("major_categories.yaml is not implemented yet")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("min_similarity", 0.35)


def test_classify_returns_one_prediction_per_review(sample_records):
    classify = get_component_function(*CLASSIFY)

    results = classify(sample_records)

    assert isinstance(results, list)
    assert len(results) == len(sample_records)


def test_major_category_output_structure(sample_records):
    classify = get_component_function(*CLASSIFY)

    results = classify(sample_records)

    for result in results:
        assert {"review_id", "predicted_major_category", "confidence"} <= set(result)
        assert result["predicted_major_category"] in MAJOR_CATEGORIES
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1


def test_every_review_is_classified_once(sample_records):
    classify = get_component_function(*CLASSIFY)

    results = classify(sample_records)

    result_ids = [r["review_id"] for r in results]
    input_ids = [r["review_id"] for r in sample_records]

    assert sorted(result_ids) == sorted(input_ids)
    assert len(result_ids) == len(set(result_ids))


def test_output_contains_no_gold_labels(sample_records):
    classify = get_component_function(*CLASSIFY)

    results = classify(sample_records)

    for result in results:
        assert not GOLD_FIELDS & set(result)


def test_below_threshold_prediction_is_other():
    classify = get_component_function(*CLASSIFY)
    threshold = _category_min_similarity()
    records = [
        {"review_id": "X1", "review_text": "Purple elephants dance quietly on Tuesdays.", "review_date": None},
        {"review_id": "X2", "review_text": "zxqv blorp wem", "review_date": None},
        {"review_id": "X3", "review_text": "ok", "review_date": None},
    ]

    results = classify(records)

    for result in results:
        if result["confidence"] < threshold:
            assert result["predicted_major_category"] == "Other"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Examples taken from config/major_categories.yaml (Docs/12).
        ("I was charged twice.", "Billing and Payments"),
        ("The OTP never arrives.", "Authentication and Account Access"),
        ("I do not receive notifications.", "Notifications"),
        ("Support never responds.", "Customer Support"),
    ],
)
def test_taxonomy_examples_are_classified_into_their_category(text, expected):
    classify = get_component_function(*CLASSIFY)

    [result] = classify([{"review_id": "E1", "review_text": text, "review_date": None}])

    assert result["predicted_major_category"] == expected
    assert result["confidence"] >= _category_min_similarity()


def test_classification_is_deterministic(sample_records):
    classify = get_component_function(*CLASSIFY)

    first = classify(sample_records)
    second = classify(sample_records)

    assert [r["predicted_major_category"] for r in first] == [r["predicted_major_category"] for r in second]
    assert [r["confidence"] for r in first] == pytest.approx([r["confidence"] for r in second], abs=1e-6)


def test_input_records_are_not_modified(sample_records):
    classify = get_component_function(*CLASSIFY)
    before = [dict(record) for record in sample_records]

    classify(sample_records)

    assert sample_records == before
