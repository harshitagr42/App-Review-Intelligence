
from conftest import get_component_function


def test_sentiment_returns_one_prediction_per_review(sample_records):
    analyze = get_component_function(
        "app_review_intelligence.components.sentiment.interface",
        "analyze",
    )

    results = analyze(sample_records)

    assert isinstance(results, list)
    assert len(results) == len(sample_records)


def test_sentiment_output_structure(sample_records):
    analyze = get_component_function(
        "app_review_intelligence.components.sentiment.interface",
        "analyze",
    )

    results = analyze(sample_records)

    valid_sentiments = {"Positive", "Negative", "Neutral"}

    for result in results:
        assert "review_id" in result
        assert "sentiment" in result
        assert "confidence" in result

        assert result["sentiment"] in valid_sentiments
        assert 0 <= result["confidence"] <= 1


def test_sentiment_does_not_receive_gold_labels(sample_records):
    analyze = get_component_function(
        "app_review_intelligence.components.sentiment.interface",
        "analyze",
    )

    # Only model-facing records are passed to the component.
    results = analyze(sample_records)

    for result in results:
        assert "gold_sentiment" not in result
