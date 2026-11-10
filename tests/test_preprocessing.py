
from conftest import get_component_function


def test_preprocessing_returns_list(sample_records):
    preprocess_records = get_component_function(
        "app_review_intelligence.preprocessing.interface",
        "preprocess_records",
    )

    result = preprocess_records(sample_records)

    assert isinstance(result, list)
    assert len(result) == len(sample_records)


def test_preprocessing_preserves_review_ids(sample_records):
    preprocess_records = get_component_function(
        "app_review_intelligence.preprocessing.interface",
        "preprocess_records",
    )

    result = preprocess_records(sample_records)

    original_ids = {r["review_id"] for r in sample_records}
    processed_ids = {r["review_id"] for r in result}

    assert original_ids == processed_ids


def test_preprocessing_does_not_remove_review_text(sample_records):
    preprocess_records = get_component_function(
        "app_review_intelligence.preprocessing.interface",
        "preprocess_records",
    )

    result = preprocess_records(sample_records)

    for record in result:
        assert "review_text" in record
        assert isinstance(record["review_text"], str)
        assert record["review_text"].strip() != ""
