"""
Tests for components/subcategory (Person 4).

Gemini is always replaced by the `fake_gemini` fixture from conftest.py, so
these tests never call the real API. The only real call is the optional
manual smoke test at the bottom (`pytest -m gemini_api`).
"""

import json
import logging
import os
import re
from collections import Counter

import pytest

from conftest import (
    CATEGORY_CODES,
    CONFIG_DIR,
    FAKE_API_KEY,
    GOLD_FIELDS,
    UNASSIGNED,
    get_component_function,
)


DISCOVER = (
    "app_review_intelligence.components.subcategory.interface",
    "discover_subcategories",
)
LLM_BATCH_SIZE = 40  # Docs/07, section 2: no more than 40 reviews per request.


def _max_retries():
    """Configured retry limit (models.llm.max_retries), default 3."""
    try:
        import yaml

        with open(CONFIG_DIR / "settings.yaml", encoding="utf-8") as handle:
            return int(yaml.safe_load(handle)["models"]["llm"]["max_retries"])
    except Exception:
        return 3


def _records(count, category="Performance and Stability", start=1001):
    return [
        {
            "review_id": f"TST{start + index}",
            "review_text": f"The app crashes every time I open screen {index}.",
            "predicted_major_category": category,
        }
        for index in range(count)
    ]


def _ids_in_prompt(prompt, records):
    return [
        record["review_id"]
        for record in records
        if re.search(rf"(?<![\w-]){re.escape(record['review_id'])}(?![\w-])", prompt)
    ]


def _assert_every_review_exactly_once(records, result):
    input_ids = [record["review_id"] for record in records]
    assigned = [assignment["review_id"] for assignment in result["assignments"]]

    assert Counter(assigned) == Counter(input_ids), "every input review must appear exactly once"

    valid_ids = {sub["subcategory_id"] for sub in result["subcategories"]}
    for assignment in result["assignments"]:
        assert assignment["subcategory_id"] == UNASSIGNED or assignment["subcategory_id"] in valid_ids


# ---------------------------------------------------------------------------
# Output format (writing_block.md 5E, Docs/05_DATA_SCHEMA.md 3E)
# ---------------------------------------------------------------------------

def test_discovery_returns_expected_structure(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(3)
    fake_gemini(records)

    result = discover(records)

    assert isinstance(result, dict)
    assert isinstance(result["subcategories"], list)
    assert isinstance(result["assignments"], list)


def test_subcategory_assignments_use_valid_review_ids(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(3)
    fake_gemini(records)

    result = discover(records)

    input_ids = {r["review_id"] for r in records}
    assigned_ids = [assignment["review_id"] for assignment in result["assignments"]]

    assert set(assigned_ids).issubset(input_ids)
    assert len(assigned_ids) == len(set(assigned_ids))


def test_every_review_is_assigned_or_explicitly_unassigned(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(3)
    fake_gemini(records)

    result = discover(records)

    _assert_every_review_exactly_once(records, result)


def test_subcategory_records_have_required_fields_and_ids(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(3) + _records(2, category="Billing and Payments", start=2001)
    fake_gemini(records)

    result = discover(records)

    assert result["subcategories"]
    seen_ids = set()
    for sub in result["subcategories"]:
        assert {"subcategory_id", "subcategory_name", "description", "major_category"} <= set(sub)
        assert sub["subcategory_name"].strip()
        assert sub["description"].strip()
        assert sub["major_category"] in CATEGORY_CODES
        # IDs are <category code>_<3-digit number>, unique within the run.
        code = CATEGORY_CODES[sub["major_category"]]
        assert re.fullmatch(rf"{code}_\d{{3}}", sub["subcategory_id"]), sub["subcategory_id"]
        assert sub["subcategory_id"] not in seen_ids
        seen_ids.add(sub["subcategory_id"])
        assert sub["subcategory_id"] != UNASSIGNED


def test_unassigned_reviews_are_kept_as_unassigned(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(4)
    fake_gemini(records, mode="unassigned")

    result = discover(records)

    _assert_every_review_exactly_once(records, result)
    by_id = {a["review_id"]: a["subcategory_id"] for a in result["assignments"]}
    assert by_id[records[-1]["review_id"]] == UNASSIGNED


# ---------------------------------------------------------------------------
# What is sent to Gemini (writing_block.md section 7, Docs/02 section 7)
# ---------------------------------------------------------------------------

def test_reviews_are_grouped_by_predicted_major_category(fake_gemini):
    discover = get_component_function(*DISCOVER)
    perf = _records(3)
    bill = _records(3, category="Billing and Payments", start=2001)
    records = perf + bill
    state = fake_gemini(records)

    discover(records)

    category_by_id = {r["review_id"]: r["predicted_major_category"] for r in records}
    sent = set()
    for prompt in state["calls"]:
        ids = _ids_in_prompt(prompt, records)
        assert len({category_by_id[i] for i in ids}) <= 1, "a request mixed major categories"
        sent.update(ids)
    assert sent == set(category_by_id)


def test_requests_contain_at_most_40_reviews(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(85)
    state = fake_gemini(records)

    result = discover(records)

    assert len(state["calls"]) >= 3
    sent = set()
    for prompt in state["calls"]:
        ids = _ids_in_prompt(prompt, records)
        assert len(ids) <= LLM_BATCH_SIZE
        sent.update(ids)
    assert sent == {r["review_id"] for r in records}
    _assert_every_review_exactly_once(records, result)


def test_prompt_has_only_id_text_and_predicted_category(fake_gemini):
    discover = get_component_function(*DISCOVER)
    # Extra fields must never reach Gemini, even if a caller passes them.
    records = [
        {**record, "review_date": "2031-02-03", "secret_column": "DO-NOT-SEND-THIS"}
        for record in _records(3)
    ]
    state = fake_gemini(records)

    discover(records)

    assert state["calls"]
    prompts = "\n".join(state["calls"])
    for record in records:
        assert record["review_text"] in prompts
    assert "Performance and Stability" in prompts
    assert "2031-02-03" not in prompts
    assert "DO-NOT-SEND-THIS" not in prompts
    for field in GOLD_FIELDS:
        assert field not in prompts
    assert FAKE_API_KEY not in prompts


# ---------------------------------------------------------------------------
# Untrusted responses (Docs/07 section 2 "Validation")
# ---------------------------------------------------------------------------

def test_malformed_json_is_retried_then_accepted(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(3)
    state = fake_gemini(records, mode="malformed", bad_calls=1)

    result = discover(records)

    assert len(state["calls"]) >= 2
    _assert_every_review_exactly_once(records, result)


def test_persistent_malformed_json_fails_with_batch_ids_after_bounded_retries(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(3)
    state = fake_gemini(records, mode="malformed")

    with pytest.raises(Exception) as error:
        discover(records)

    assert len(state["calls"]) <= 1 + _max_retries()
    assert any(record["review_id"] in str(error.value) for record in records)


@pytest.mark.parametrize("mode", ["invented_id", "duplicate_id", "missing_id", "wrong_category"])
def test_invalid_responses_never_corrupt_assignments(fake_gemini, mode):
    discover = get_component_function(*DISCOVER)
    records = _records(4)
    state = fake_gemini(records, mode=mode)

    # The component may fail after its bounded retries, or recover by
    # marking reviews UNASSIGNED. It must never accept bad data silently.
    try:
        result = discover(records)
    except Exception:
        assert len(state["calls"]) <= 1 + _max_retries()
        return

    _assert_every_review_exactly_once(records, result)
    assert "INVENTED_999" not in json.dumps(result)


def test_bad_response_is_retried_and_recovers(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(4)
    state = fake_gemini(records, mode="invented_id", bad_calls=1)

    result = discover(records)

    assert len(state["calls"]) >= 2
    _assert_every_review_exactly_once(records, result)
    assert "INVENTED_999" not in json.dumps(result)


# ---------------------------------------------------------------------------
# Cross-batch consolidation (Docs/07 section 2)
# ---------------------------------------------------------------------------

@pytest.mark.local_model
def test_same_issue_from_different_batches_is_consolidated(fake_gemini):
    discover = get_component_function(*DISCOVER)
    records = _records(80)  # two batches, both name the group "Application Crashes"
    state = fake_gemini(records)

    result = discover(records)

    assert len(state["calls"]) >= 2
    perf = [s for s in result["subcategories"] if s["major_category"] == "Performance and Stability"]
    assert len(perf) == 1
    assert perf[0]["subcategory_id"] == "PERF_001"
    _assert_every_review_exactly_once(records, result)


def test_subcategories_are_never_merged_across_major_categories(fake_gemini):
    discover = get_component_function(*DISCOVER)
    # Both categories receive an identically named group from the fake.
    records = _records(3) + _records(3, category="Billing and Payments", start=2001)
    fake_gemini(records)

    result = discover(records)

    category_by_id = {r["review_id"]: r["predicted_major_category"] for r in records}
    sub_category = {s["subcategory_id"]: s["major_category"] for s in result["subcategories"]}
    assert set(sub_category.values()) == {"Performance and Stability", "Billing and Payments"}
    for assignment in result["assignments"]:
        if assignment["subcategory_id"] != UNASSIGNED:
            assert sub_category[assignment["subcategory_id"]] == category_by_id[assignment["review_id"]]


# ---------------------------------------------------------------------------
# API key handling (Docs/02 section 8)
# ---------------------------------------------------------------------------

def test_missing_api_key_stops_before_any_request(fake_gemini, monkeypatch):
    discover = get_component_function(*DISCOVER)
    records = _records(3)
    state = fake_gemini(records)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(Exception):
        discover(records)

    assert state["calls"] == []


def test_api_key_is_never_logged_or_returned(fake_gemini, caplog):
    discover = get_component_function(*DISCOVER)
    records = _records(3)
    fake_gemini(records, mode="malformed", bad_calls=1)

    with caplog.at_level(logging.DEBUG):
        result = discover(records)

    assert FAKE_API_KEY not in caplog.text
    assert FAKE_API_KEY not in json.dumps(result)


# ---------------------------------------------------------------------------
# Optional manual smoke test against the real API
# ---------------------------------------------------------------------------

@pytest.mark.gemini_api
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
def test_real_gemini_smoke():
    """Run explicitly with `pytest -m gemini_api`. Sends three synthetic reviews."""
    discover = get_component_function(*DISCOVER)
    records = [
        {"review_id": "S1", "review_text": "The app crashes when I open it.",
         "predicted_major_category": "Performance and Stability"},
        {"review_id": "S2", "review_text": "It closes suddenly on startup.",
         "predicted_major_category": "Performance and Stability"},
        {"review_id": "S3", "review_text": "Loading the home screen takes forever.",
         "predicted_major_category": "Performance and Stability"},
    ]

    result = discover(records)

    _assert_every_review_exactly_once(records, result)
