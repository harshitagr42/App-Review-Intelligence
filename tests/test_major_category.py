
from conftest import get_component_function


def test_classify_returns_one_prediction_per_review(sample_records):
    classify = get_component_function(
        "app_review_intelligence.components.major_category.interface",
        "classify",
    )

    results = classify(sample_records)

    assert isinstance(results, list)
    assert len(results) == len(sample_records)


def test_major_category_output_structure(sample_records):
    classify = get_component_function(
        "app_review_intelligence.components.major_category.interface",
        "classify",
    )

    results = classify(sample_records)

    valid_categories = {
        "Performance and Stability",
        "Features and Functionality",
        "Billing and Payments",
        "Authentication and Account Access",
        "User Interface and Experience",
        "Customer Support",
        "Privacy and Security",
        "Notifications",
        "Compatibility",
        "Other",
    }

    for result in results:
        assert "review_id" in result
        assert "predicted_major_category" in result
        assert "confidence" in result

        assert result["predicted_major_category"] in valid_categories
        assert 0 <= result["confidence"] <= 1


def test_every_review_is_classified_once(sample_records):
    classify = get_component_function(
        "app_review_intelligence.components.major_category.interface",
        "classify",
    )

    results = classify(sample_records)

    result_ids = [r["review_id"] for r in results]
    input_ids = [r["review_id"] for r in sample_records]

    assert sorted(result_ids) == sorted(input_ids)
    assert len(result_ids) == len(set(result_ids))
