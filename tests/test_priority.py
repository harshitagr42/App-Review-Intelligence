"""
Tests for components/priority (Person 5).

Formula and bands: Docs/07_SUBCATEGORY_AND_PRIORITY_RULES.md, section 4.
Output format: writing_block.md 5F plus the `components` object from
Docs/05_DATA_SCHEMA.md 3F.

`calculate_priorities(subcategories, sentiment_results)` receives the full
subcategory result (writing_block.md 5E: `subcategories` + `assignments`),
because review counts come from the assignments.
"""

import copy
import math

import pytest

from conftest import UNASSIGNED, get_component_function


CALCULATE = (
    "app_review_intelligence.components.priority.interface",
    "calculate_priorities",
)
BASE_WEIGHTS = {"volume": 0.35, "negative_sentiment": 0.35, "severity": 0.20, "trend": 0.10}
COMPONENT_KEYS = {
    "volume",
    "negative_sentiment",
    "severity",
    "trend",
    "effective_weights",
    "severity_reason_codes",
    "trend_status",
}


def _level_for(score):
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _build(groups):
    """
    groups: list of (subcategory_id, major_category, [(review_id, sentiment), ...]).
    Returns (subcategory_result, sentiment_results).
    """
    subcategories, assignments, sentiments = [], [], []
    for subcategory_id, major_category, reviews in groups:
        if subcategory_id != UNASSIGNED:
            subcategories.append({
                "subcategory_id": subcategory_id,
                "subcategory_name": f"Issue {subcategory_id}",
                "description": f"Reviews about issue {subcategory_id}.",
                "major_category": major_category,
            })
        for review_id, sentiment in reviews:
            assignments.append({"review_id": review_id, "subcategory_id": subcategory_id})
            sentiments.append({"review_id": review_id, "sentiment": sentiment, "confidence": 0.9})
    return {"subcategories": subcategories, "assignments": assignments}, sentiments


def _reviews(prefix, negative=0, neutral=0, positive=0):
    labels = ["Negative"] * negative + ["Neutral"] * neutral + ["Positive"] * positive
    return [(f"{prefix}{i}", label) for i, label in enumerate(labels)]


def _by_id(results):
    return {result["subcategory_id"]: result for result in results}


def test_priority_returns_one_result_per_subcategory():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("PERF_001", "Performance and Stability", _reviews("a", negative=2)),
        ("PERF_002", "Performance and Stability", _reviews("b", positive=1)),
    ])

    result = calculate_priorities(subcategories, sentiments)

    assert isinstance(result, list)
    assert sorted(r["subcategory_id"] for r in result) == ["PERF_001", "PERF_002"]


def test_priority_output_structure():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("PERF_001", "Performance and Stability", _reviews("a", negative=1, positive=1)),
    ])

    [result] = calculate_priorities(subcategories, sentiments)

    for field in ("subcategory_id", "subcategory_name", "review_count", "negative_proportion",
                  "priority_score", "priority_level", "explanation", "components"):
        assert field in result
    assert result["subcategory_name"] == "Issue PERF_001"
    assert 0 <= result["negative_proportion"] <= 1
    assert 0 <= result["priority_score"] <= 100
    assert result["priority_level"] in {"Critical", "High", "Medium", "Low"}
    assert isinstance(result["explanation"], str) and result["explanation"].strip()
    assert COMPONENT_KEYS <= set(result["components"])
    assert result["components"]["severity"] in {25, 50, 75, 100}


def test_review_count_and_negative_proportion():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("BILL_001", "Billing and Payments", _reviews("a", negative=2, positive=1)),
    ])

    [result] = calculate_priorities(subcategories, sentiments)

    assert result["review_count"] == 3
    # Full precision is kept; rounding is for display only.
    assert result["negative_proportion"] == pytest.approx(2 / 3, abs=1e-9)
    assert result["components"]["negative_sentiment"] == pytest.approx(100 * 2 / 3, abs=1e-6)


def test_volume_is_log_scaled_within_each_major_category():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("PERF_001", "Performance and Stability", _reviews("a", negative=9)),
        ("PERF_002", "Performance and Stability", _reviews("b", negative=3)),
        ("BILL_001", "Billing and Payments", _reviews("c", negative=1)),
    ])

    results = _by_id(calculate_priorities(subcategories, sentiments))

    assert results["PERF_001"]["components"]["volume"] == pytest.approx(100.0, abs=1e-6)
    expected = 100 * math.log(1 + 3) / math.log(1 + 9)
    assert results["PERF_002"]["components"]["volume"] == pytest.approx(expected, abs=1e-6)
    # Largest (only) group in its own major category.
    assert results["BILL_001"]["components"]["volume"] == pytest.approx(100.0, abs=1e-6)


def test_missing_trend_redistributes_weights():
    calculate_priorities = get_component_function(*CALCULATE)
    # No review dates are available, so trend must be unavailable.
    subcategories, sentiments = _build([
        ("PERF_001", "Performance and Stability", _reviews("a", negative=3)),
    ])

    [result] = calculate_priorities(subcategories, sentiments)
    components = result["components"]
    weights = components["effective_weights"]

    assert components["trend"] is None
    assert components["trend_status"]
    assert weights["trend"] == pytest.approx(0.0)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    for key in ("volume", "negative_sentiment", "severity"):
        assert weights[key] == pytest.approx(BASE_WEIGHTS[key] / 0.9, abs=1e-3)


def test_score_is_weighted_sum_of_components_and_level_matches_band():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("PERF_001", "Performance and Stability", _reviews("a", negative=5, positive=1)),
        ("PERF_002", "Performance and Stability", _reviews("b", negative=1, neutral=1)),
        ("UIUX_001", "User Interface and Experience", _reviews("c", positive=4)),
    ])

    for result in calculate_priorities(subcategories, sentiments):
        components = result["components"]
        weights = components["effective_weights"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
        expected = sum(
            weights[key] * (components[key] or 0)
            for key in ("volume", "negative_sentiment", "severity", "trend")
        )
        expected = min(max(expected, 0.0), 100.0)
        assert result["priority_score"] == pytest.approx(expected, abs=0.01)
        assert result["priority_level"] == _level_for(result["priority_score"])


def test_positive_high_volume_group_is_not_critical():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("FEAT_001", "Features and Functionality", _reviews("p", positive=100)),
        ("FEAT_002", "Features and Functionality", _reviews("n", negative=2)),
    ])

    results = _by_id(calculate_priorities(subcategories, sentiments))

    assert results["FEAT_001"]["components"]["negative_sentiment"] == pytest.approx(0.0)
    assert results["FEAT_001"]["priority_level"] != "Critical"


def test_unassigned_reviews_are_excluded_from_ranking():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("PERF_001", "Performance and Stability", _reviews("a", negative=2)),
        (UNASSIGNED, None, _reviews("u", negative=50)),
    ])

    results = calculate_priorities(subcategories, sentiments)

    assert [r["subcategory_id"] for r in results] == ["PERF_001"]
    # UNASSIGNED reviews must not count toward any group's volume.
    assert results[0]["review_count"] == 2
    assert results[0]["components"]["volume"] == pytest.approx(100.0, abs=1e-6)


def test_priorities_are_deterministic_and_inputs_unchanged():
    calculate_priorities = get_component_function(*CALCULATE)
    subcategories, sentiments = _build([
        ("PERF_001", "Performance and Stability", _reviews("a", negative=4, positive=2)),
        ("BILL_001", "Billing and Payments", _reviews("b", negative=1, neutral=2)),
    ])
    subcategories_before = copy.deepcopy(subcategories)
    sentiments_before = copy.deepcopy(sentiments)

    first = calculate_priorities(subcategories, sentiments)
    second = calculate_priorities(subcategories, sentiments)

    assert first == second
    assert subcategories == subcategories_before
    assert sentiments == sentiments_before
