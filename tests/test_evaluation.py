
from conftest import get_component_function


def test_evaluation_returns_metrics(sample_labels):
    evaluate = get_component_function(
        "app_review_intelligence.evaluation.interface",
        "evaluate_predictions",
    )

    major = [
        {"review_id": "R0001", "predicted_major_category": "Performance and Stability"},
        {"review_id": "R0002", "predicted_major_category": "Usability"},
        {"review_id": "R0003", "predicted_major_category": "Performance and Stability"},
    ]

    sentiment = [
        {"review_id": "R0001", "sentiment": "Negative"},
        {"review_id": "R0002", "sentiment": "Positive"},
        {"review_id": "R0003", "sentiment": "Negative"},
    ]

    assignments = [
        {"review_id": "R0001", "subcategory_id": "PERF_001"},
        {"review_id": "R0002", "subcategory_id": "USE_001"},
        {"review_id": "R0003", "subcategory_id": "PERF_002"},
    ]

    result = evaluate(
        major,
        sentiment,
        assignments,
        sample_labels,
    )

    assert isinstance(result, dict)
    assert "major_category" in result
    assert "sentiment" in result
    assert "subcategory" in result


def test_evaluation_does_not_claim_priority_accuracy(sample_labels):
    evaluate = get_component_function(
        "app_review_intelligence.evaluation.interface",
        "evaluate_predictions",
    )

    major = [
        {"review_id": "R0001", "predicted_major_category": "Performance and Stability"},
        {"review_id": "R0002", "predicted_major_category": "Usability"},
        {"review_id": "R0003", "predicted_major_category": "Performance and Stability"},
    ]

    sentiment = [
        {"review_id": "R0001", "sentiment": "Negative"},
        {"review_id": "R0002", "sentiment": "Positive"},
        {"review_id": "R0003", "sentiment": "Negative"},
    ]

    assignments = [
        {"review_id": "R0001", "subcategory_id": "PERF_001"},
        {"review_id": "R0002", "subcategory_id": "USE_001"},
        {"review_id": "R0003", "subcategory_id": "PERF_002"},
    ]

    result = evaluate(major, sentiment, assignments, sample_labels)

    assert "priority_accuracy" not in result
