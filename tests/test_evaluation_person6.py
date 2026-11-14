
import pytest

from app_review_intelligence.evaluation.interface import (
    evaluate_predictions,
)


def make_predictions():
    # Deliberately shuffled to verify matching by review_id.
    major = [
        {"review_id": "R0003", "predicted_major_category": "Performance and Stability"},
        {"review_id": "R0001", "predicted_major_category": "Performance and Stability"},
        {"review_id": "R0002", "predicted_major_category": "Usability"},
    ]

    sentiment = [
        {"review_id": "R0002", "sentiment": "Positive"},
        {"review_id": "R0003", "sentiment": "Negative"},
        {"review_id": "R0001", "sentiment": "Negative"},
    ]

    assignments = [
        {"review_id": "R0001", "subcategory_id": "PERF_001"},
        {"review_id": "R0003", "subcategory_id": "PERF_002"},
        {"review_id": "R0002", "subcategory_id": "USE_001"},
    ]

    return major, sentiment, assignments


def make_labels():
    return [
        {
            "review_id": "R0001",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "App Crashes",
            "gold_sentiment": "Negative",
        },
        {
            "review_id": "R0002",
            "gold_major_category": "Usability",
            "gold_minor_category": "Easy to Use",
            "gold_sentiment": "Positive",
        },
        {
            "review_id": "R0003",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "Slow Performance",
            "gold_sentiment": "Negative",
        },
    ]


def test_evaluation_matches_reviews_by_id():
    major, sentiment, assignments = make_predictions()

    result = evaluate_predictions(
        major, sentiment, assignments, make_labels()
    )

    assert result["evaluation_summary"]["labelled_review_count"] == 3
    assert result["evaluation_summary"]["matching_method"] == "review_id"
    assert result["major_category"]["accuracy"] == 1.0
    assert result["sentiment"]["accuracy"] == 1.0
    assert result["subcategory"]["adjusted_rand_index"] == 1.0


def test_evaluation_rejects_missing_prediction():
    major, sentiment, assignments = make_predictions()
    major.pop()

    with pytest.raises(ValueError, match="review IDs do not match"):
        evaluate_predictions(
            major, sentiment, assignments, make_labels()
        )


def test_evaluation_rejects_duplicate_review_ids():
    major, sentiment, assignments = make_predictions()
    major.append(major[0].copy())

    with pytest.raises(ValueError, match="Duplicate review IDs"):
        evaluate_predictions(
            major, sentiment, assignments, make_labels()
        )


def test_evaluation_does_not_report_priority_accuracy():
    major, sentiment, assignments = make_predictions()

    result = evaluate_predictions(
        major, sentiment, assignments, make_labels()
    )

    assert "priority_accuracy" not in result
    assert "priorities" not in result
