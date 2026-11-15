"""
Tests for components/sentiment (Person 3).

Uses cardiffnlp/twitter-roberta-base-sentiment-latest locally
(Docs/06_MODEL_SPECIFICATION.md, section 3).
"""

import pytest

from conftest import GOLD_FIELDS, SENTIMENT_LABELS, get_component_function


pytestmark = pytest.mark.local_model

ANALYZE = (
    "app_review_intelligence.components.sentiment.interface",
    "analyze",
)


def test_sentiment_returns_one_prediction_per_review(sample_records):
    analyze = get_component_function(*ANALYZE)

    results = analyze(sample_records)

    assert isinstance(results, list)
    assert len(results) == len(sample_records)


def test_sentiment_output_structure(sample_records):
    analyze = get_component_function(*ANALYZE)

    results = analyze(sample_records)

    for result in results:
        assert {"review_id", "sentiment", "confidence"} <= set(result)
        assert result["sentiment"] in SENTIMENT_LABELS
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1


def test_every_review_gets_exactly_one_sentiment(sample_records):
    analyze = get_component_function(*ANALYZE)

    results = analyze(sample_records)

    result_ids = [r["review_id"] for r in results]
    assert sorted(result_ids) == sorted(r["review_id"] for r in sample_records)
    assert len(result_ids) == len(set(result_ids))


def test_sentiment_does_not_receive_gold_labels(sample_records):
    analyze = get_component_function(*ANALYZE)

    # Only model-facing records are passed to the component.
    results = analyze(sample_records)

    for result in results:
        assert not GOLD_FIELDS & set(result)


def test_final_partial_batch_is_processed():
    analyze = get_component_function(*ANALYZE)
    # Batch size is 8, so 11 records leave a final batch of 3.
    records = [
        {"review_id": f"B{i}", "review_text": f"Review number {i} about the app.", "review_date": None}
        for i in range(11)
    ]

    results = analyze(records)

    assert sorted(r["review_id"] for r in results) == sorted(r["review_id"] for r in records)


def test_emoji_text_is_handled():
    analyze = get_component_function(*ANALYZE)
    records = [{"review_id": "E1", "review_text": "Love it 😍😍 best app ever!!", "review_date": None}]

    [result] = analyze(records)

    assert result["sentiment"] == "Positive"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I love this app, it is perfect and works great!", "Positive"),
        ("This is the worst app ever. It keeps crashing and I hate it.", "Negative"),
        ("I don't like this app at all, it is terrible.", "Negative"),
    ],
)
def test_clear_examples(text, expected):
    analyze = get_component_function(*ANALYZE)

    [result] = analyze([{"review_id": "C1", "review_text": text, "review_date": None}])

    assert result["sentiment"] == expected


def test_sentiment_is_deterministic(sample_records):
    analyze = get_component_function(*ANALYZE)

    first = analyze(sample_records)
    second = analyze(sample_records)

    assert [r["sentiment"] for r in first] == [r["sentiment"] for r in second]
    assert [r["confidence"] for r in first] == pytest.approx([r["confidence"] for r in second], abs=1e-6)
