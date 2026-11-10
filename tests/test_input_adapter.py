
import pandas as pd
import pytest

from conftest import get_component_function


def test_load_and_prepare_returns_three_outputs(tmp_path):
    load_and_prepare = get_component_function(
        "app_review_intelligence.adapters.input.interface",
        "load_and_prepare",
    )

    file_path = tmp_path / "reviews.csv"

    df = pd.DataFrame([
        {
            "ID": "R0001",
            "Review": "The application crashes.",
            "Date": "2026-07-21",
            "Major Class": "Performance and Stability",
            "Minor Class": "App Crashes",
            "Sentiment": "Negative",
        }
    ])

    df.to_csv(file_path, index=False)

    model_records, evaluation_labels, metadata = load_and_prepare(
        [str(file_path)]
    )

    assert isinstance(model_records, list)
    assert isinstance(evaluation_labels, list)
    assert isinstance(metadata, dict)

    assert len(model_records) == 1
    assert len(evaluation_labels) == 1


def test_ground_truth_is_not_in_model_records(tmp_path):
    load_and_prepare = get_component_function(
        "app_review_intelligence.adapters.input.interface",
        "load_and_prepare",
    )

    file_path = tmp_path / "reviews.csv"

    pd.DataFrame([{
        "ID": "R0001",
        "Review": "Very slow application.",
        "Date": "2026-07-21",
        "Major Class": "Performance and Stability",
        "Minor Class": "Slow Performance",
        "Sentiment": "Negative",
    }]).to_csv(file_path, index=False)

    model_records, evaluation_labels, _ = load_and_prepare(
        [str(file_path)]
    )

    assert "review_text" in model_records[0]
    assert "review_id" in model_records[0]

    assert "gold_major_category" not in model_records[0]
    assert "gold_minor_category" not in model_records[0]
    assert "gold_sentiment" not in model_records[0]

    assert "gold_major_category" in evaluation_labels[0]
    assert "gold_minor_category" in evaluation_labels[0]
    assert "gold_sentiment" in evaluation_labels[0]


def test_duplicate_review_ids_are_rejected_or_corrected(tmp_path):
    load_and_prepare = get_component_function(
        "app_review_intelligence.adapters.input.interface",
        "load_and_prepare",
    )

    file_path = tmp_path / "reviews.csv"

    pd.DataFrame([
        {
            "ID": "R0001",
            "Review": "First review",
            "Date": "2026-07-21",
            "Major Class": "Usability",
            "Minor Class": "Navigation",
            "Sentiment": "Positive",
        },
        {
            "ID": "R0001",
            "Review": "Second review",
            "Date": "2026-07-22",
            "Major Class": "Usability",
            "Minor Class": "Navigation",
            "Sentiment": "Positive",
        },
    ]).to_csv(file_path, index=False)

    try:
        records, _, _ = load_and_prepare([str(file_path)])
    except ValueError:
        return

    ids = [record["review_id"] for record in records]
    assert len(ids) == len(set(ids))
