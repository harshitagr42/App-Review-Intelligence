"""Extra tests for components/priority: severity matching, trend, optional arguments, errors."""

import pytest

from conftest import get_component_function

CALCULATE = (
    "app_review_intelligence.components.priority.interface",
    "calculate_priorities",
)


def _subs(*items):
    return {
        "subcategories": [
            {"subcategory_id": i, "subcategory_name": n, "description": d, "major_category": "Authentication and Account Access"}
            for i, n, d in items
        ],
        "assignments": [],
    }


def _assign(result, subcategory_id, review_ids):
    result["assignments"].extend({"review_id": r, "subcategory_id": subcategory_id} for r in review_ids)


def _sent(review_ids, label="Negative"):
    return [{"review_id": r, "sentiment": label, "confidence": 0.9} for r in review_ids]


def test_severity_from_subcategory_text_without_records():
    calculate = get_component_function(*CALCULATE)
    result = _subs(("AUTH_001", "Cannot Log In", "Users who can't login."))
    _assign(result, "AUTH_001", ["a", "b"])

    [row] = calculate(result, _sent(["a", "b"]))

    assert row["components"]["severity"] == 75
    assert row["components"]["severity_reason_codes"] == ["high:cannot log in"]
    assert row["components"]["severity_source"] == "subcategory_text"


def test_severity_uses_highest_level_from_review_text():
    calculate = get_component_function(*CALCULATE)
    result = _subs(("AUTH_001", "Account problems", "Misc."))
    _assign(result, "AUTH_001", ["a", "b"])
    records = [
        {"review_id": "a", "review_text": "I cannot login at all", "review_date": "2026-07-01"},
        {"review_id": "b", "review_text": "Possible data breach on my account!", "review_date": "2026-07-02"},
    ]

    [row] = calculate(result, _sent(["a", "b"]), review_records=records)

    assert row["components"]["severity"] == 100
    assert row["components"]["severity_reason_codes"][0] == "critical:data breach"
    assert "high:cannot log in" in row["components"]["severity_reason_codes"]
    assert row["components"]["severity_source"] == "subcategory_and_review_text"


def test_no_severity_phrase_gives_baseline():
    calculate = get_component_function(*CALCULATE)
    result = _subs(("AUTH_001", "Misc", "Misc."))
    _assign(result, "AUTH_001", ["a"])

    [row] = calculate(result, _sent(["a"]))

    assert row["components"]["severity"] == 25
    assert row["components"]["severity_reason_codes"] == ["no_phrase_match"]


def _dated_records(prefix, dates):
    return [{"review_id": f"{prefix}{i}", "review_text": "meh", "review_date": d} for i, d in enumerate(dates)]


def test_trend_rising_is_computed_from_dates():
    calculate = get_component_function(*CALCULATE)
    recent = _dated_records("r", ["2026-07-30"] * 6)       # last 30 days
    previous = _dated_records("p", ["2026-06-20"] * 5)     # 30 days before that
    ids = [r["review_id"] for r in recent + previous]
    result = _subs(("AUTH_001", "Misc", "Misc."))
    _assign(result, "AUTH_001", ids)
    sentiments = _sent([r["review_id"] for r in recent]) + _sent([r["review_id"] for r in previous], "Positive")

    [row] = calculate(result, sentiments, review_records=recent + previous)

    assert row["components"]["trend"] == pytest.approx(100.0)
    assert row["components"]["trend_status"].startswith("computed")
    assert row["components"]["effective_weights"]["trend"] == pytest.approx(0.10)
    assert "complaints rising" in row["explanation"]


def test_trend_unavailable_with_too_few_reviews_per_window():
    calculate = get_component_function(*CALCULATE)
    records = _dated_records("a", ["2026-07-30", "2026-06-20"])
    result = _subs(("AUTH_001", "Misc", "Misc."))
    _assign(result, "AUTH_001", ["a0", "a1"])

    [row] = calculate(result, _sent(["a0", "a1"]), review_records=records)

    assert row["components"]["trend"] is None
    assert row["components"]["trend_status"].startswith("insufficient_data")
    assert row["components"]["effective_weights"]["trend"] == 0.0


def test_results_sorted_highest_score_first():
    calculate = get_component_function(*CALCULATE)
    result = _subs(("A_001", "Big and bad", "x"), ("A_002", "Small and happy", "x"))
    _assign(result, "A_001", ["a", "b", "c", "d"])
    _assign(result, "A_002", ["e"])
    sentiments = _sent(["a", "b", "c", "d"]) + _sent(["e"], "Positive")

    scores = [r["priority_score"] for r in calculate(result, sentiments)]

    assert scores == sorted(scores, reverse=True)


def test_config_override_changes_bands():
    calculate = get_component_function(*CALCULATE)
    result = _subs(("A_001", "Misc", "x"))
    _assign(result, "A_001", ["a"])

    [row] = calculate(result, _sent(["a"]), config={"priority": {"priority_bands": {"critical": 40, "high": 30, "medium": 20}}})

    assert row["priority_level"] == "Critical"


@pytest.mark.parametrize("mutate", ["duplicate_review", "unknown_subcategory", "bad_sentiment"])
def test_invalid_input_raises_value_error(mutate):
    calculate = get_component_function(*CALCULATE)
    result = _subs(("A_001", "Misc", "x"))
    _assign(result, "A_001", ["a"])
    sentiments = _sent(["a"])
    if mutate == "duplicate_review":
        _assign(result, "A_001", ["a"])
    elif mutate == "unknown_subcategory":
        _assign(result, "NOPE_001", ["b"])
    else:
        sentiments = [{"review_id": "a", "sentiment": "Angry", "confidence": 0.5}]

    with pytest.raises(ValueError):
        calculate(result, sentiments)


def test_malformed_subcategory_result_raises_type_error():
    calculate = get_component_function(*CALCULATE)
    with pytest.raises(TypeError):
        calculate([], [])
