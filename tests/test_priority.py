from conftest import get_component_function


def _subcategory_result(review_ids):
    """Shared format E: discovered subcategories plus the review-ID assignments."""
    return {
        "subcategories": [
            {
                "subcategory_id": "PERF_001",
                "subcategory_name": "App Crashes",
                "description": "Application unexpectedly crashes.",
                "major_category": "Performance and Stability",
            }
        ],
        "assignments": [
            {"review_id": review_id, "subcategory_id": "PERF_001"}
            for review_id in review_ids
        ],
    }


def test_priority_returns_list():
    calculate_priorities = get_component_function(
        "app_review_intelligence.components.priority.interface",
        "calculate_priorities",
    )

    sentiment_results = [
        {"review_id": "1", "sentiment": "Negative", "confidence": 0.95},
        {"review_id": "2", "sentiment": "Negative", "confidence": 0.90},
    ]

    result = calculate_priorities(_subcategory_result(["1", "2"]), sentiment_results)

    assert isinstance(result, list)
    assert len(result) == 1


def test_priority_output_structure():
    calculate_priorities = get_component_function(
        "app_review_intelligence.components.priority.interface",
        "calculate_priorities",
    )

    sentiments = [
        {"review_id": "1", "sentiment": "Negative", "confidence": 0.9},
        {"review_id": "2", "sentiment": "Positive", "confidence": 0.9},
    ]

    results = calculate_priorities(_subcategory_result(["1", "2"]), sentiments)

    result = results[0]

    assert "subcategory_id" in result
    assert "review_count" in result
    assert "negative_proportion" in result
    assert "priority_score" in result
    assert "priority_level" in result
    assert "explanation" in result

    assert result["subcategory_id"] == "PERF_001"
    assert result["review_count"] == 2
    assert result["negative_proportion"] == 0.5  # 1 negative of 2 reviews

    assert 0 <= result["negative_proportion"] <= 1
    assert 0 <= result["priority_score"] <= 100
    assert result["priority_level"] in {
        "Critical", "High", "Medium", "Low"
    }
