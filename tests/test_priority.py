
from conftest import get_component_function


def test_priority_returns_list():
    calculate_priorities = get_component_function(
        "app_review_intelligence.components.priority.interface",
        "calculate_priorities",
    )

    subcategories = [
        {
            "subcategory_id": "PERF_001",
            "subcategory_name": "Application Crashes",
            "description": "Application unexpectedly crashes.",
        }
    ]

    sentiment_results = [
        {
            "review_id": "R0001",
            "sentiment": "Negative",
            "confidence": 0.95,
        },
        {
            "review_id": "R0002",
            "sentiment": "Negative",
            "confidence": 0.90,
        },
    ]

    result = calculate_priorities(
        subcategories,
        sentiment_results,
    )

    assert isinstance(result, list)
    assert len(result) == 1


def test_priority_output_structure():
    calculate_priorities = get_component_function(
        "app_review_intelligence.components.priority.interface",
        "calculate_priorities",
    )

    subcategories = [{
        "subcategory_id": "PERF_001",
        "subcategory_name": "Application Crashes",
        "description": "Application unexpectedly crashes.",
    }]

    sentiments = [
        {"review_id": "R0001", "sentiment": "Negative", "confidence": 0.9},
        {"review_id": "R0002", "sentiment": "Positive", "confidence": 0.9},
    ]

    results = calculate_priorities(subcategories, sentiments)

    result = results[0]

    assert "subcategory_id" in result
    assert "review_count" in result
    assert "negative_proportion" in result
    assert "priority_score" in result
    assert "priority_level" in result
    assert "explanation" in result

    assert 0 <= result["negative_proportion"] <= 1
    assert 0 <= result["priority_score"] <= 100
    assert result["priority_level"] in {
        "Critical", "High", "Medium", "Low"
    }
