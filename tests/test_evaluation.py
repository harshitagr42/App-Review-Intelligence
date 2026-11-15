"""
Tests for evaluation (Person 6). Metrics: Docs/08_EVALUATION_PLAN.md.

Call shape used by these tests:
    evaluate_predictions(major_predictions, sentiment_predictions,
                         subcategory_assignments, evaluation_labels) -> dict

Metric key names are not fixed by the docs; the names these tests expect are
listed once in METRIC_KEYS so the team can agree on them in one place.
"""

import json

import pytest

from conftest import UNASSIGNED, get_component_function


EVALUATE = ("app_review_intelligence.evaluation.interface", "evaluate_predictions")

METRIC_KEYS = {
    "major_category": {"accuracy", "macro_f1", "weighted_f1"},
    "sentiment": {"accuracy", "macro_f1", "weighted_f1"},
    "subcategory": {
        "ari", "ami", "nmi", "v_measure", "homogeneity", "completeness",
        "coverage", "unassigned_ratio",
    },
}

GOLD = [
    # review_id, gold major, gold minor, gold sentiment (lowercase, as in the dataset file)
    ("1", "Performance and Stability", "App Crashes", "negative"),
    ("2", "Performance and Stability", "App Crashes", "negative"),
    ("3", "User Interface and Experience", "Navigation Confusion", "neutral"),
    ("4", "Billing and Payments", "Duplicate Charges", "negative"),
    ("5", "Billing and Payments", "Duplicate Charges", "negative"),
    ("6", "Billing and Payments", "Refund Delays", "positive"),
]
PRED_MAJOR = [
    "Performance and Stability",
    "User Interface and Experience",
    "User Interface and Experience",
    "Billing and Payments",
    "Billing and Payments",
    "Performance and Stability",
]
PRED_SENTIMENT = ["Negative", "Negative", "Negative", "Negative", "Neutral", "Positive"]
PRED_SUBCATEGORY = ["PERF_001", "PERF_001", "UIUX_001", "BILL_001", "BILL_001", "BILL_002"]


def _inputs(subcategory_ids=PRED_SUBCATEGORY):
    labels = [
        {"review_id": rid, "gold_major_category": major, "gold_minor_category": minor, "gold_sentiment": sent}
        for rid, major, minor, sent in GOLD
    ]
    major = [
        {"review_id": rid, "predicted_major_category": pred, "confidence": 0.8}
        for (rid, *_), pred in zip(GOLD, PRED_MAJOR)
    ]
    sentiment = [
        {"review_id": rid, "sentiment": pred, "confidence": 0.9}
        for (rid, *_), pred in zip(GOLD, PRED_SENTIMENT)
    ]
    assignments = [
        {"review_id": rid, "subcategory_id": sub}
        for (rid, *_), sub in zip(GOLD, subcategory_ids)
    ]
    return major, sentiment, assignments, labels


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_evaluation_returns_metrics(sample_labels):
    evaluate = get_component_function(*EVALUATE)

    major = [
        {"review_id": "1", "predicted_major_category": "Performance and Stability"},
        {"review_id": "2", "predicted_major_category": "User Interface and Experience"},
        {"review_id": "3", "predicted_major_category": "Performance and Stability"},
    ]
    sentiment = [
        {"review_id": "1", "sentiment": "Negative"},
        {"review_id": "2", "sentiment": "Positive"},
        {"review_id": "3", "sentiment": "Negative"},
    ]
    assignments = [
        {"review_id": "1", "subcategory_id": "PERF_001"},
        {"review_id": "2", "subcategory_id": "UIUX_001"},
        {"review_id": "3", "subcategory_id": "PERF_002"},
    ]

    result = evaluate(major, sentiment, assignments, sample_labels)

    assert isinstance(result, dict)
    for section, keys in METRIC_KEYS.items():
        assert section in result
        assert keys <= set(result[section]), f"{section} is missing {keys - set(result[section])}"


def test_major_category_metrics_on_known_fixture():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    evaluate = get_component_function(*EVALUATE)
    gold = [g[1] for g in GOLD]

    result = evaluate(*_inputs())["major_category"]

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx(
        sklearn_metrics.f1_score(gold, PRED_MAJOR, average="macro", zero_division=0)
    )
    assert result["weighted_f1"] == pytest.approx(
        sklearn_metrics.f1_score(gold, PRED_MAJOR, average="weighted", zero_division=0)
    )


def test_sentiment_metrics_are_case_insensitive_on_known_fixture():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    evaluate = get_component_function(*EVALUATE)
    gold = [g[3].capitalize() for g in GOLD]

    result = evaluate(*_inputs())["sentiment"]

    # Gold labels are lowercase; predictions are Title case. 4 of 6 match.
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx(
        sklearn_metrics.f1_score(gold, PRED_SENTIMENT, average="macro", zero_division=0)
    )


def test_subcategory_clustering_metrics_on_known_fixture():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    evaluate = get_component_function(*EVALUATE)
    gold = [g[2] for g in GOLD]

    result = evaluate(*_inputs())["subcategory"]

    assert result["ari"] == pytest.approx(sklearn_metrics.adjusted_rand_score(gold, PRED_SUBCATEGORY))
    assert result["ami"] == pytest.approx(sklearn_metrics.adjusted_mutual_info_score(gold, PRED_SUBCATEGORY))
    assert result["nmi"] == pytest.approx(sklearn_metrics.normalized_mutual_info_score(gold, PRED_SUBCATEGORY))
    assert result["v_measure"] == pytest.approx(sklearn_metrics.v_measure_score(gold, PRED_SUBCATEGORY))
    assert result["coverage"] == pytest.approx(1.0)
    assert result["unassigned_ratio"] == pytest.approx(0.0)


def test_subcategory_metrics_are_never_called_accuracy():
    evaluate = get_component_function(*EVALUATE)

    result = evaluate(*_inputs())

    assert not any("accuracy" in key.lower() for key in _all_keys(result["subcategory"]))


def test_subcategory_is_scored_with_clustering_metrics_not_accuracy():
    evaluate = get_component_function(*EVALUATE)

    result = evaluate(*_inputs())

    # Discovered subcategories have no fixed names, so they are scored with
    # clustering metrics (ARI/AMI/NMI/V-measure) and never called "accuracy".
    assert "accuracy" not in result["subcategory"]


def test_unassigned_reviews_are_counted():
    evaluate = get_component_function(*EVALUATE)
    with_unassigned = list(PRED_SUBCATEGORY[:4]) + [UNASSIGNED, UNASSIGNED]

    result = evaluate(*_inputs(with_unassigned))["subcategory"]

    assert result["coverage"] == pytest.approx(4 / 6)
    assert result["unassigned_ratio"] == pytest.approx(2 / 6)


def test_evaluation_does_not_claim_priority_accuracy(sample_labels):
    evaluate = get_component_function(*EVALUATE)

    result = evaluate(*_inputs())

    assert "priority_accuracy" not in result
    for key in _all_keys(result):
        lowered = key.lower()
        if "priority" in lowered:
            assert not any(word in lowered for word in ("accuracy", "precision", "recall", "f1"))


def test_metrics_change_with_predictions():
    """Metrics are computed from the inputs, not hardcoded."""
    evaluate = get_component_function(*EVALUATE)
    major, sentiment, assignments, labels = _inputs()
    perfect_major = [
        {"review_id": label["review_id"], "predicted_major_category": label["gold_major_category"], "confidence": 0.9}
        for label in labels
    ]

    imperfect = evaluate(major, sentiment, assignments, labels)["major_category"]["accuracy"]
    perfect = evaluate(perfect_major, sentiment, assignments, labels)["major_category"]["accuracy"]

    assert perfect == pytest.approx(1.0)
    assert imperfect < perfect


def test_evaluation_result_is_valid_json():
    evaluate = get_component_function(*EVALUATE)

    result = evaluate(*_inputs())

    # No NaN/Infinity: evaluation.json must be strict JSON.
    json.dumps(result, allow_nan=False)
