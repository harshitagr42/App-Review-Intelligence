"""Comprehensive unit and internal tests for components/subcategory (Person 4).

Covers response_parser, prompt_builder, gemini_client, implementation orchestration,
and consolidator (both offline with stub vectors and with the local MiniLM model).
"""

import copy
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pytest

from app_review_intelligence.components.subcategory.consolidator import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MERGE_SIMILARITY,
    UNASSIGNED,
    consolidate_subcategories,
    embed_texts,
    normalize_name,
)
from app_review_intelligence.components.subcategory.gemini_client import (
    DEFAULT_API_KEY_ENV_VAR,
    check_api_key,
    generate_content,
    is_transient_error,
)
from app_review_intelligence.components.subcategory.implementation import (
    _sleep,
    load_category_codes,
    load_settings,
    run_discovery,
)
from app_review_intelligence.components.subcategory.interface import (
    discover_subcategories,
    discover_subcategories_with_report,
)
from app_review_intelligence.components.subcategory.prompt_builder import (
    build_retry_hint,
    build_user_prompt,
    load_prompts,
)
from app_review_intelligence.components.subcategory.response_parser import (
    parse_response,
    repair,
    validate_against_batch,
)
from conftest import FAKE_API_KEY


# ===========================================================================
# 1. response_parser tests
# ===========================================================================


def test_valid_json_parses():
    raw = json.dumps({
        "major_category": "Performance and Stability",
        "subcategories": [
            {
                "subcategory_name": "App Crashes",
                "description": "Reviews reporting unexpected shutdowns.",
                "review_ids": ["R1", "R2"],
            }
        ],
        "unassigned_review_ids": ["R3"],
    })
    parsed = parse_response(raw)
    assert parsed["major_category"] == "Performance and Stability"
    assert len(parsed["subcategories"]) == 1
    assert parsed["subcategories"][0]["subcategory_name"] == "App Crashes"
    assert parsed["subcategories"][0]["review_ids"] == ["R1", "R2"]
    assert parsed["unassigned_review_ids"] == ["R3"]


def test_json_inside_code_fences_parses():
    raw = (
        "```json\n"
        + json.dumps({
            "major_category": "Billing and Payments",
            "subcategories": [
                {
                    "subcategory_name": "Payment Failure",
                    "description": "Payment gateway errors.",
                    "review_ids": ["B1"],
                }
            ],
            "unassigned_review_ids": [],
        })
        + "\n```"
    )
    parsed = parse_response(raw)
    assert parsed["major_category"] == "Billing and Payments"
    assert parsed["subcategories"][0]["subcategory_name"] == "Payment Failure"


def test_non_json_raises_value_error():
    with pytest.raises(ValueError, match="Failed to parse response as JSON"):
        parse_response("Sorry, here are your categories: not valid json")


@pytest.mark.parametrize(
    "bad_payload",
    [
        "not a dict",
        [],
        {"subcategories": []},  # missing major_category
        {"major_category": 123, "subcategories": []},  # wrong typed major_category
        {"major_category": "Other"},  # missing subcategories
        {"major_category": "Other", "subcategories": "not a list"},
        {"major_category": "Other", "subcategories": ["not a dict"]},
        {"major_category": "Other", "subcategories": [{"description": "d", "review_ids": []}]},  # missing name
        {"major_category": "Other", "subcategories": [{"subcategory_name": "", "description": "d", "review_ids": []}]},  # empty name
        {"major_category": "Other", "subcategories": [{"subcategory_name": "N", "review_ids": []}]},  # missing desc
        {"major_category": "Other", "subcategories": [{"subcategory_name": "N", "description": "", "review_ids": []}]},  # empty desc
        {"major_category": "Other", "subcategories": [{"subcategory_name": "N", "description": "d"}]},  # missing review_ids
        {"major_category": "Other", "subcategories": [{"subcategory_name": "N", "description": "d", "review_ids": "not list"}]},
        {"major_category": "Other", "subcategories": [], "unassigned_review_ids": "not a list"},
    ],
)
def test_missing_or_wrong_typed_fields_raise_value_error(bad_payload):
    raw = json.dumps(bad_payload) if not isinstance(bad_payload, str) else bad_payload
    with pytest.raises(ValueError):
        parse_response(raw)


def test_model_supplied_subcategory_id_is_ignored():
    raw = json.dumps({
        "major_category": "Performance and Stability",
        "subcategories": [
            {
                "subcategory_id": "MODEL_LOCAL_999",
                "subcategory_name": "Crash on Startup",
                "description": "App crashes immediately.",
                "review_ids": ["R1"],
            }
        ],
    })
    parsed = parse_response(raw)
    assert "subcategory_id" not in parsed["subcategories"][0]


def test_missing_unassigned_review_ids_defaults_to_empty_list():
    raw = json.dumps({
        "major_category": "Other",
        "subcategories": [
            {
                "subcategory_name": "General Feedback",
                "description": "Feedback reviews.",
                "review_ids": ["R1"],
            }
        ],
    })
    parsed = parse_response(raw)
    assert parsed["unassigned_review_ids"] == []


def test_validate_against_batch_flags_all_problem_types():
    # 1. Wrong category
    p1 = {
        "major_category": "Wrong Category",
        "subcategories": [{"subcategory_name": "A", "description": "d", "review_ids": ["R1", "R2"]}],
        "unassigned_review_ids": [],
    }
    assert "wrong_category" in validate_against_batch(p1, ["R1", "R2"], "Performance and Stability")

    # 2. Invented IDs
    p2 = {
        "major_category": "Performance and Stability",
        "subcategories": [{"subcategory_name": "A", "description": "d", "review_ids": ["R1", "INVENTED_999"]}],
        "unassigned_review_ids": [],
    }
    assert "invented_ids" in validate_against_batch(p2, ["R1"], "Performance and Stability")

    # 3. Duplicate IDs across subcategories
    p3 = {
        "major_category": "Performance and Stability",
        "subcategories": [
            {"subcategory_name": "A", "description": "d", "review_ids": ["R1"]},
            {"subcategory_name": "B", "description": "d", "review_ids": ["R1"]},
        ],
        "unassigned_review_ids": [],
    }
    assert "duplicate_ids" in validate_against_batch(p3, ["R1"], "Performance and Stability")

    # 4. Duplicate IDs across subcategory and unassigned
    p4 = {
        "major_category": "Performance and Stability",
        "subcategories": [{"subcategory_name": "A", "description": "d", "review_ids": ["R1"]}],
        "unassigned_review_ids": ["R1"],
    }
    assert "duplicate_ids" in validate_against_batch(p4, ["R1"], "Performance and Stability")

    # 5. Missing IDs
    p5 = {
        "major_category": "Performance and Stability",
        "subcategories": [{"subcategory_name": "A", "description": "d", "review_ids": ["R1"]}],
        "unassigned_review_ids": [],
    }
    assert "missing_ids" in validate_against_batch(p5, ["R1", "R2"], "Performance and Stability")


def test_validate_against_batch_empty_review_ids_dropped_not_flagged():
    parsed = {
        "major_category": "Performance and Stability",
        "subcategories": [
            {"subcategory_name": "A", "description": "d", "review_ids": ["R1"]},
            {"subcategory_name": "EmptyGroup", "description": "d", "review_ids": []},
        ],
        "unassigned_review_ids": [],
    }
    problems = validate_against_batch(parsed, ["R1"], "Performance and Stability")
    assert problems == []
    # Verify the empty subcategory was dropped
    assert len(parsed["subcategories"]) == 1
    assert parsed["subcategories"][0]["subcategory_name"] == "A"


def test_repair_cleans_defects_and_preserves_input():
    parsed = {
        "major_category": "Performance and Stability",
        "subcategories": [
            {
                "subcategory_name": "Group A",
                "description": "Desc A",
                "review_ids": ["R1", "INVENTED_1", "R2", "R1"],  # invented & duplicate
            },
            {
                "subcategory_name": "Group Emptied",
                "description": "Desc Empty",
                "review_ids": ["INVENTED_2"],  # will become empty
            },
        ],
        "unassigned_review_ids": ["R2", "INVENTED_3"],  # R2 duplicate, INVENTED_3 invented
    }
    parsed_copy = copy.deepcopy(parsed)
    batch_ids = ["R1", "R2", "R3"]  # R3 is missing

    repaired, changes = repair(parsed, batch_ids)

    # Input dict must not be mutated
    assert parsed == parsed_copy

    # Group Emptied was dropped
    assert len(repaired["subcategories"]) == 1
    assert repaired["subcategories"][0]["subcategory_name"] == "Group A"
    assert repaired["subcategories"][0]["review_ids"] == ["R1", "R2"]

    # Missing R3 moved into unassigned
    assert repaired["unassigned_review_ids"] == ["R3"]

    # Change log contains recorded problems
    problems_logged = {c["problem"] for c in changes}
    assert "invented_id" in problems_logged
    assert "duplicate_id" in problems_logged
    assert "missing_id" in problems_logged


# ===========================================================================
# 2. prompt_builder tests
# ===========================================================================


def test_build_user_prompt_replaces_placeholders_and_preserves_json_braces():
    template = (
        'Category: {{MAJOR_CATEGORY}}\n'
        'Reviews: {{REVIEWS_JSON}}\n'
        'Schema: {"example": {"code": "VAL", "ids": ["ID_A"]}}'
    )
    records = [
        {"review_id": "R100", "review_text": "Crashes on screen A {with curly braces}."}
    ]
    prompt = build_user_prompt(template, "Performance and Stability", records)

    assert "Performance and Stability" in prompt
    assert "R100" in prompt
    assert "Crashes on screen A {with curly braces}." in prompt
    assert '{"example": {"code": "VAL", "ids": ["ID_A"]}}' in prompt
    assert "{{MAJOR_CATEGORY}}" not in prompt
    assert "{{REVIEWS_JSON}}" not in prompt


def test_prompts_have_no_dollar_signs():
    system_text, user_text = load_prompts()
    assert "$" not in system_text, "System prompt contains '$'"
    assert "$" not in user_text, "User prompt contains '$'"


def test_load_prompts_raises_file_not_found_for_missing_or_empty_file(tmp_path):
    # Missing files
    with pytest.raises(FileNotFoundError, match="System prompt file not found"):
        load_prompts(tmp_path)

    # Empty system prompt
    sys_file = tmp_path / "subcategory_system.txt"
    sys_file.write_text("   ", encoding="utf-8")
    user_file = tmp_path / "subcategory_user_template.txt"
    user_file.write_text("template content", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="is empty"):
        load_prompts(tmp_path)

    # Empty user prompt
    sys_file.write_text("system content", encoding="utf-8")
    user_file.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="is empty"):
        load_prompts(tmp_path)


def test_build_retry_hint_lists_only_batch_ids():
    # Pass missing IDs that include an invented ID
    summary = {"missing_ids": ["R1", "INVENTED_999"], "batch_ids": ["R1", "R2"]}
    hint = build_retry_hint(summary)

    assert "R1" in hint
    assert "INVENTED_999" not in hint

    # When explicit batch_ids is provided as second arg
    hint2 = build_retry_hint({"missing_ids": ["R2", "INV_ABC"]}, batch_ids=["R2"])
    assert "R2" in hint2
    assert "INV_ABC" not in hint2


# ===========================================================================
# 3. gemini_client tests
# ===========================================================================


def test_check_api_key_raises_value_error_without_leaking_key(monkeypatch):
    monkeypatch.delenv("TEST_KEY_ENV", raising=False)
    with pytest.raises(ValueError) as err:
        check_api_key("TEST_KEY_ENV")
    assert "TEST_KEY_ENV" in str(err.value)
    assert FAKE_API_KEY not in str(err.value)

    monkeypatch.setenv("TEST_KEY_ENV", "   ")
    with pytest.raises(ValueError):
        check_api_key("TEST_KEY_ENV")


def test_is_transient_error():
    # Helper class representing mock exceptions
    class MockStatusError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    class MockCodeError(Exception):
        def __init__(self, code):
            self.code = code

    class MockStringStatusError(Exception):
        def __init__(self, status):
            self.status = status

    # Transient errors
    assert is_transient_error(MockStatusError(429))
    assert is_transient_error(MockStatusError(500))
    assert is_transient_error(MockStatusError(503))
    assert is_transient_error(MockCodeError(429))
    assert is_transient_error(MockCodeError(502))
    assert is_transient_error(MockStringStatusError("RESOURCE_EXHAUSTED"))
    assert is_transient_error(TimeoutError("Request timed out"))
    assert is_transient_error(ConnectionError("Connection reset by peer"))

    # Permanent client errors
    assert not is_transient_error(MockStatusError(400))
    assert not is_transient_error(MockStatusError(401))
    assert not is_transient_error(MockStatusError(403))
    assert not is_transient_error(MockStatusError(404))
    assert not is_transient_error(MockCodeError(400))
    assert not is_transient_error(MockStringStatusError("INVALID_ARGUMENT"))


# ===========================================================================
# 4. implementation tests
# ===========================================================================


def test_chunking_85_reviews_produces_expected_batches(fake_gemini, monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation._sleep", sleep_calls.append)

    records = [
        {
            "review_id": f"R{i:04d}",
            "review_text": f"App crashes on tap {i}.",
            "predicted_major_category": "Performance and Stability",
        }
        for i in range(85)
    ]
    state = fake_gemini(records)

    result, report = discover_subcategories_with_report(records)

    assert report["batches"] == 3
    # Check that batches contain 40, 40, 5
    batch_sizes = [
        len([r for r in records if r["review_id"] in call])
        for call in state["calls"]
    ]
    assert batch_sizes == [40, 40, 5]


def test_transient_error_then_success_backoff_and_capped_sleep(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)
    sleep_calls = []
    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation._sleep", sleep_calls.append)

    call_count = [0]

    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            class Mock503(Exception):
                code = 503
            raise Mock503("Temporary service outage")
        return json.dumps({
            "major_category": "Performance and Stability",
            "subcategories": [
                {
                    "subcategory_name": "Crash",
                    "description": "App crashes.",
                    "review_ids": ["R1"],
                }
            ],
            "unassigned_review_ids": [],
        })

    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation.generate_content", mock_generate)

    records = [
        {
            "review_id": "R1",
            "review_text": "Crashes on login.",
            "predicted_major_category": "Performance and Stability",
        }
    ]

    result, report = discover_subcategories_with_report(records)

    assert call_count[0] == 2
    assert report["retries"] == 1
    assert len(sleep_calls) == 1
    assert 0 < sleep_calls[0] <= 60.0  # At or below cap


def test_permanent_error_raises_immediately_no_retry(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)
    sleep_calls = []
    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation._sleep", sleep_calls.append)

    call_count = [0]

    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        class Mock400(Exception):
            code = 400
        raise Mock400("Bad Request: invalid payload")

    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation.generate_content", mock_generate)

    records = [
        {
            "review_id": "R1",
            "review_text": "Crashes on login.",
            "predicted_major_category": "Performance and Stability",
        }
    ]

    with pytest.raises(Exception) as exc:
        discover_subcategories(records)

    assert "Bad Request" in str(exc.value)
    assert call_count[0] == 1  # No retry
    assert len(sleep_calls) == 0


def test_total_calls_never_exceed_max_attempts_mixed_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation._sleep", lambda s: None)

    call_count = [0]

    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            class Mock503(Exception):
                code = 503
            raise Mock503("503 Server Error")
        return "Not valid JSON {"

    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation.generate_content", mock_generate)

    records = [
        {
            "review_id": "R1",
            "review_text": "Crashes on login.",
            "predicted_major_category": "Performance and Stability",
        }
    ]

    # max_retries default is 3 -> max_attempts = 4
    with pytest.raises(RuntimeError):
        discover_subcategories(records, config={"max_retries": 3})

    assert call_count[0] == 4


def test_first_call_temp_0_retry_temp_0_3(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation._sleep", lambda s: None)

    temperatures = []

    def mock_generate(system_prompt, user_prompt, temperature=0.0, **kwargs):
        temperatures.append(temperature)
        if len(temperatures) == 1:
            return "Invalid JSON"
        return json.dumps({
            "major_category": "Performance and Stability",
            "subcategories": [
                {"subcategory_name": "Crash", "description": "Desc", "review_ids": ["R1"]}
            ],
            "unassigned_review_ids": [],
        })

    monkeypatch.setattr("app_review_intelligence.components.subcategory.implementation.generate_content", mock_generate)

    records = [
        {"review_id": "R1", "review_text": "Crashes.", "predicted_major_category": "Performance and Stability"}
    ]
    discover_subcategories(records)

    assert temperatures == [0.0, 0.3]


@pytest.mark.parametrize(
    ("bad_input", "error_type", "match_str"),
    [
        ("not a list", ValueError, "records must be a list"),
        ([{"not_a_dict": True}], ValueError, "missing 'review_id'"),
        ([{"review_id": "", "review_text": "t", "predicted_major_category": "Other"}], ValueError, "invalid or missing 'review_id'"),
        ([{"review_id": "R1", "review_text": "  ", "predicted_major_category": "Other"}], ValueError, "empty or non-string 'review_text'"),
        ([
            {"review_id": "R1", "review_text": "t", "predicted_major_category": "Other"},
            {"review_id": "R1", "review_text": "t2", "predicted_major_category": "Other"},
        ], ValueError, "Duplicate review_id"),
        ([{"review_id": "R1", "review_text": "t", "predicted_major_category": "NoSuchCategory"}], ValueError, "Unknown predicted_major_category"),
    ],
)
def test_invalid_inputs_raise_exact_exceptions(bad_input, error_type, match_str, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)
    with pytest.raises(error_type, match=match_str):
        discover_subcategories(bad_input)


def test_determinism_and_assignment_order(fake_gemini):
    records = [
        {"review_id": "R3", "review_text": "Crash 3", "predicted_major_category": "Performance and Stability"},
        {"review_id": "R1", "review_text": "Crash 1", "predicted_major_category": "Billing and Payments"},
        {"review_id": "R2", "review_text": "Crash 2", "predicted_major_category": "Performance and Stability"},
    ]
    fake_gemini(records)

    res1 = discover_subcategories(records)
    res2 = discover_subcategories(records)

    assert res1 == res2
    assert [a["review_id"] for a in res1["assignments"]] == ["R3", "R1", "R2"]


def test_report_serializable_and_contains_no_secrets(fake_gemini):
    records = [
        {"review_id": "R1", "review_text": "Secret review text content.", "predicted_major_category": "Other"}
    ]
    fake_gemini(records)

    result, report = discover_subcategories_with_report(records)

    serialized = json.dumps(report)
    assert FAKE_API_KEY not in serialized
    assert "Secret review text" not in serialized
    assert report["batches"] == 1
    assert "categories" in report


# ===========================================================================
# 5. consolidator tests
# ===========================================================================


def test_exact_name_merge_ignores_case_punctuation_spaces():
    raw = [
        {"subcategory_name": "App Crashes", "description": "Desc 1", "review_ids": ["R1"]},
        {"subcategory_name": "app  crashes!!", "description": "Desc 2", "review_ids": ["R2"]},
        {"subcategory_name": "  APP CRASHES.  ", "description": "Desc 3", "review_ids": ["R3"]},
    ]
    subs, mapping, merges = consolidate_subcategories(
        "Performance and Stability", "PERF", raw
    )

    assert len(subs) == 1
    assert subs[0]["subcategory_id"] == "PERF_001"
    assert subs[0]["subcategory_name"] == "App Crashes"
    assert mapping == {"R1": "PERF_001", "R2": "PERF_001", "R3": "PERF_001"}
    assert merges == 2


def test_semantic_merge_with_stub_vectors(monkeypatch):
    # A and B have cosine >= 0.82; C is orthogonal
    # Vector A: [1.0, 0.0]
    # Vector B: [0.9, 0.43588989] -> dot with A is 0.90 >= 0.82
    # Vector C: [0.0, 1.0] -> dot with A is 0.0 < 0.82
    stub_vectors = {
        "A": np.array([1.0, 0.0], dtype=np.float32),
        "B": np.array([0.9, 0.43588989], dtype=np.float32),
        "C": np.array([0.0, 1.0], dtype=np.float32),
    }

    def mock_embed(texts, model_id=DEFAULT_EMBEDDING_MODEL):
        result = []
        for t in texts:
            key = t.split(":")[0]
            result.append(stub_vectors[key])
        return np.vstack(result)

    monkeypatch.setattr("app_review_intelligence.components.subcategory.consolidator.embed_texts", mock_embed)

    raw = [
        {"subcategory_name": "A", "description": "Desc A", "review_ids": ["R1", "R2"]},
        {"subcategory_name": "B", "description": "Desc B", "review_ids": ["R3"]},
        {"subcategory_name": "C", "description": "Desc C", "review_ids": ["R4"]},
    ]

    subs, mapping, merges = consolidate_subcategories(
        "Performance and Stability", "PERF", raw, merge_similarity=0.82
    )

    assert len(subs) == 2
    # A has 2 reviews, B has 1 -> A is representative, B joins A
    group_a = next(s for s in subs if s["subcategory_name"] == "A")
    group_c = next(s for s in subs if s["subcategory_name"] == "C")

    assert mapping["R1"] == group_a["subcategory_id"]
    assert mapping["R2"] == group_a["subcategory_id"]
    assert mapping["R3"] == group_a["subcategory_id"]
    assert mapping["R4"] == group_c["subcategory_id"]
    assert merges == 1


def test_semantic_merge_avoids_chaining(monkeypatch):
    # Test that candidate C does not join cluster A even if C is close to B,
    # because C must be compared to cluster A's representative!
    # A is representative: [1.0, 0.0]
    # B: [0.85, 0.52678] -> dot with A is 0.85 (joins A)
    # C: [0.55, 0.83516] -> dot with A is 0.55 (< 0.82!), but dot with B is ~0.906 (> 0.82)
    stub_vectors = {
        "Issue A": np.array([1.0, 0.0], dtype=np.float32),
        "Issue B": np.array([0.85, 0.5267828], dtype=np.float32),
        "Issue C": np.array([0.55, 0.83516465], dtype=np.float32),
    }

    def mock_embed(texts, model_id=DEFAULT_EMBEDDING_MODEL):
        result = []
        for t in texts:
            key = t.split(":")[0]
            result.append(stub_vectors[key])
        return np.vstack(result)

    monkeypatch.setattr("app_review_intelligence.components.subcategory.consolidator.embed_texts", mock_embed)

    raw = [
        {"subcategory_name": "Issue A", "description": "Desc A", "review_ids": ["R1", "R2", "R3"]},
        {"subcategory_name": "Issue B", "description": "Desc B", "review_ids": ["R4", "R5"]},
        {"subcategory_name": "Issue C", "description": "Desc C", "review_ids": ["R6"]},
    ]

    subs, mapping, merges = consolidate_subcategories(
        "Performance and Stability", "PERF", raw, merge_similarity=0.82
    )

    # Issue C must NOT join Issue A, so there should be 2 subcategories: Issue A (with B merged) and Issue C
    assert len(subs) == 2
    names = {s["subcategory_name"] for s in subs}
    assert names == {"Issue A", "Issue C"}
    assert mapping["R6"] != mapping["R1"]


def test_groups_from_different_major_categories_never_merged(fake_gemini):
    records = [
        {"review_id": "R1", "review_text": "Crashes.", "predicted_major_category": "Performance and Stability"},
        {"review_id": "R2", "review_text": "Crashes.", "predicted_major_category": "Billing and Payments"},
    ]
    fake_gemini(records)

    result = discover_subcategories(records)

    sub_categories = {s["subcategory_id"]: s["major_category"] for s in result["subcategories"]}
    assert len(sub_categories) == 2
    for a in result["assignments"]:
        expected_cat = "Performance and Stability" if a["review_id"] == "R1" else "Billing and Payments"
        assert sub_categories[a["subcategory_id"]] == expected_cat


def test_final_ids_format_ordering_and_uniqueness():
    raw = [
        {"subcategory_name": "Small Group", "description": "Desc", "review_ids": ["R1"]},
        {"subcategory_name": "Large Group", "description": "Desc", "review_ids": ["R2", "R3", "R4"]},
        {"subcategory_name": "Medium Group", "description": "Desc", "review_ids": ["R5", "R6"]},
    ]
    # No semantic merge (similarity 1.0)
    subs, mapping, merges = consolidate_subcategories(
        "Performance and Stability", "PERF", raw, merge_similarity=1.0
    )

    assert [s["subcategory_name"] for s in subs] == ["Large Group", "Medium Group", "Small Group"]
    assert [s["subcategory_id"] for s in subs] == ["PERF_001", "PERF_002", "PERF_003"]


def test_embed_texts_call_counts(monkeypatch):
    embed_calls = []

    def mock_embed(texts, model_id=DEFAULT_EMBEDDING_MODEL):
        embed_calls.append(texts)
        return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr("app_review_intelligence.components.subcategory.consolidator.embed_texts", mock_embed)

    # 1 group -> 0 embed calls
    consolidate_subcategories(
        "Performance and Stability",
        "PERF",
        [{"subcategory_name": "A", "description": "d", "review_ids": ["R1"]}],
    )
    assert len(embed_calls) == 0

    # 2 identical names after exact-name merge -> 1 group remaining -> 0 embed calls
    consolidate_subcategories(
        "Performance and Stability",
        "PERF",
        [
            {"subcategory_name": "A", "description": "d", "review_ids": ["R1"]},
            {"subcategory_name": "a", "description": "d", "review_ids": ["R2"]},
        ],
    )
    assert len(embed_calls) == 0

    # 2 distinct names -> exactly 1 embed call
    consolidate_subcategories(
        "Performance and Stability",
        "PERF",
        [
            {"subcategory_name": "A", "description": "d", "review_ids": ["R1"]},
            {"subcategory_name": "B", "description": "d", "review_ids": ["R2"]},
        ],
    )
    assert len(embed_calls) == 1


# ===========================================================================
# 6. Real MiniLM clustering test (local_model)
# ===========================================================================


@pytest.mark.local_model
def test_real_minilm_clustering_and_similarity_matrix():
    groups = [
        {
            "subcategory_name": "App Crashes On Launch",
            "description": "The app crashes immediately when opened by the user.",
            "review_ids": ["R1", "R2", "R3"],
        },
        {
            "subcategory_name": "Application Crashes",
            "description": "The application unexpectedly shuts down during usage.",
            "review_ids": ["R4", "R5"],
        },
        {
            "subcategory_name": "Unexpected App Shutdowns",
            "description": "The mobile app abruptly closes while running in the foreground.",
            "review_ids": ["R6"],
        },
        {
            "subcategory_name": "Slow Loading Screens",
            "description": "The user interface takes too long to load data and display pages.",
            "review_ids": ["R7", "R8"],
        },
    ]

    texts = [f"{g['subcategory_name']}: {g['description']}" for g in groups]
    embeddings = embed_texts(texts, model_id=DEFAULT_EMBEDDING_MODEL)

    sim_matrix = np.dot(embeddings, embeddings.T)

    # Pairwise similarities
    sim_0_1 = float(sim_matrix[0, 1])  # App Crashes On Launch vs Application Crashes
    sim_0_2 = float(sim_matrix[0, 2])  # App Crashes On Launch vs Unexpected App Shutdowns
    sim_1_2 = float(sim_matrix[1, 2])  # Application Crashes vs Unexpected App Shutdowns

    sim_0_3 = float(sim_matrix[0, 3])  # Crash vs Slow Loading
    sim_1_3 = float(sim_matrix[1, 3])  # Crash vs Slow Loading
    sim_2_3 = float(sim_matrix[2, 3])  # Crash vs Slow Loading

    # Print / record similarity values for report
    print(f"\n[MiniLM Pairwise Cosine Similarities]")
    print(f"Crashes On Launch vs Application Crashes: {sim_0_1:.4f}")
    print(f"Crashes On Launch vs Unexpected Shutdowns: {sim_0_2:.4f}")
    print(f"Application Crashes vs Unexpected Shutdowns: {sim_1_2:.4f}")
    print(f"Crashes On Launch vs Slow Loading: {sim_0_3:.4f}")
    print(f"Application Crashes vs Slow Loading: {sim_1_3:.4f}")
    print(f"Unexpected Shutdowns vs Slow Loading: {sim_2_3:.4f}")

    # Assert that Slow Loading Screens stays completely separate
    assert sim_0_3 < 0.35
    assert sim_1_3 < 0.35
    assert sim_2_3 < 0.35

    # Run actual consolidation at default threshold 0.82
    subs, mapping, merges = consolidate_subcategories(
        "Performance and Stability", "PERF", groups, merge_similarity=DEFAULT_MERGE_SIMILARITY
    )

    # Check that Slow Loading Screens is in its own subcategory
    loading_sub_id = mapping["R7"]
    assert mapping["R8"] == loading_sub_id
    assert mapping["R1"] != loading_sub_id
    assert mapping["R4"] != loading_sub_id
    assert mapping["R6"] != loading_sub_id

    # The crash groups similarities are ~0.75, ~0.69, ~0.69 (all < 0.82), so they remain separate at 0.82 threshold
    assert sim_0_1 < DEFAULT_MERGE_SIMILARITY
    assert sim_0_2 < DEFAULT_MERGE_SIMILARITY
    assert sim_1_2 < DEFAULT_MERGE_SIMILARITY
