
from conftest import get_component_function


def test_evaluation_returns_metrics(sample_labels):
    evaluate = get_component_function(
        "app_review_intelligence.evaluation.interface",
        "evaluate_predictions",
    )

    major = [
        {"review_id": "1", "predicted_major_category": "Performance and Stability"},
        {"review_id": "2", "predicted_major_category": "Other"},
        {"review_id": "3", "predicted_major_category": "Performance and Stability"},
    ]

    sentiment = [
        {"review_id": "1", "sentiment": "Negative"},
        {"review_id": "2", "sentiment": "Positive"},
        {"review_id": "3", "sentiment": "Negative"},
    ]

    assignments = [
        {"review_id": "1", "subcategory_id": "PERF_001"},
        {"review_id": "2", "subcategory_id": "OTH_001"},
        {"review_id": "3", "subcategory_id": "PERF_002"},
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
        {"review_id": "1", "predicted_major_category": "Performance and Stability"},
        {"review_id": "2", "predicted_major_category": "Other"},
        {"review_id": "3", "predicted_major_category": "Performance and Stability"},
    ]

    sentiment = [
        {"review_id": "1", "sentiment": "Negative"},
        {"review_id": "2", "sentiment": "Positive"},
        {"review_id": "3", "sentiment": "Negative"},
    ]

    assignments = [
        {"review_id": "1", "subcategory_id": "PERF_001"},
        {"review_id": "2", "subcategory_id": "OTH_001"},
        {"review_id": "3", "subcategory_id": "PERF_002"},
    ]

    result = evaluate(major, sentiment, assignments, sample_labels)

    assert "priority_accuracy" not in result


def test_subcategory_is_scored_with_clustering_metrics_not_accuracy(sample_labels):
    evaluate = get_component_function(
        "app_review_intelligence.evaluation.interface",
        "evaluate_predictions",
    )

    major = [
        {"review_id": "1", "predicted_major_category": "Performance and Stability"},
        {"review_id": "2", "predicted_major_category": "Other"},
        {"review_id": "3", "predicted_major_category": "Performance and Stability"},
    ]
    sentiment = [
        {"review_id": "1", "sentiment": "Negative"},
        {"review_id": "2", "sentiment": "Positive"},
        {"review_id": "3", "sentiment": "Negative"},
    ]
    assignments = [
        {"review_id": "1", "subcategory_id": "PERF_001"},
        {"review_id": "2", "subcategory_id": "OTH_001"},
        {"review_id": "3", "subcategory_id": "PERF_002"},
    ]

    result = evaluate(major, sentiment, assignments, sample_labels)

    # Discovered subcategories have no fixed names, so they are scored with
    # clustering metrics (ARI/AMI/NMI/V-measure) and never called "accuracy".
    assert "accuracy" not in result["subcategory"]
