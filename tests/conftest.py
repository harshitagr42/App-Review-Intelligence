import importlib
import json
import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "app_review_intelligence"
CONFIG_DIR = REPO_ROOT / "config"
PROMPTS_DIR = REPO_ROOT / "prompts"
COMBINED_DATASET = REPO_ROOT / "dataset" / "processed" / "combined_reviews_output.csv"

# Default taxonomy (Docs/05_DATA_SCHEMA.md, section 2). Names match the
# dataset's `Major Class` labels exactly; codes prefix subcategory IDs.
CATEGORY_CODES = {
    "Authentication and Account Access": "AUTH",
    "Billing and Payments": "BILL",
    "Compatibility": "COMP",
    "Customer Support": "SUPP",
    "Features and Functionality": "FEAT",
    "Notifications": "NOTI",
    "Other": "OTHR",
    "Performance and Stability": "PERF",
    "Privacy and Security": "PRIV",
    "User Interface and Experience": "UIUX",
}
MAJOR_CATEGORIES = (
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
)
SENTIMENTS = ("Positive", "Negative", "Neutral")
SENTIMENT_LABELS = set(SENTIMENTS)
GOLD_FIELDS = {"gold_major_category", "gold_minor_category", "gold_sentiment"}
UNASSIGNED = "UNASSIGNED"
FAKE_API_KEY = "test-key-NOT-A-REAL-KEY-7f3a9c"


@pytest.fixture
def sample_records():
    return [
        {
            "review_id": "1",
            "review_text": "The app keeps crashing every time I open my timetable. Really frustrating.",
            "review_date": "2026-07-21",
        },
        {
            "review_id": "2",
            "review_text": "Excellent app. Very easy to use.",
            "review_date": "2026-07-22",
        },
        {
            "review_id": "3",
            "review_text": "Since the latest update, loading is extremely slow. Please fix this.",
            "review_date": "2026-07-23",
        },
    ]


@pytest.fixture
def sample_labels():
    return [
        {
            "review_id": "1",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "App Crashes",
            "gold_sentiment": "Negative",
        },
        {
            "review_id": "2",
            "gold_major_category": "Other",
            "gold_minor_category": "General Praise",
            "gold_sentiment": "Positive",
        },
        {
            "review_id": "3",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "Slow Loading",
            "gold_sentiment": "Negative",
        },
    ]


def _import_or_skip(module_path):
    """Import a module; skip only when it (or a parent package) is missing."""
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        # Skip only when the requested component or one of its
        # parent packages is missing.
        if error.name and (
            module_path == error.name
            or module_path.startswith(error.name + ".")
        ):
            pytest.skip(f"{module_path} is not implemented yet")
        raise


def get_component_function(module_path, function_name):
    """
    Import a component function.

    If the component or function has not been implemented yet,
    skip its tests instead of failing the entire test suite.

    If the component exists but has a real implementation error,
    the test should fail normally.
    """
    module = _import_or_skip(module_path)

    function = getattr(module, function_name, None)

    if function is None:
        pytest.skip(
            f"{function_name} has not been implemented in {module_path}"
        )

    return function


# ---------------------------------------------------------------------------
# Fake Gemini client for subcategory tests
# ---------------------------------------------------------------------------
#
# Unit tests must never call the real Gemini API
# (Docs/11_TEST_AND_ACCEPTANCE_PLAN.md, section 1). The fake replaces ONE
# test seam in the subcategory component:
#
#     components/subcategory/gemini_client.py :: generate_content(...)
#
# Contract for that function: it performs one Gemini request and returns the
# raw response text (a JSON string, or invalid text when Gemini misbehaves).
# Its prompt arguments may be strings, lists or dicts; the fake reads every
# string inside them. Retries, validation and parsing happen in the code that
# calls it, so the fake exercises them. If the team renames the seam, change
# GEMINI_SEAM below.
#
# The fake answers in the shape required by
# Docs/07_SUBCATEGORY_AND_PRIORITY_RULES.md, section 2.

GEMINI_SEAM = (
    "app_review_intelligence.components.subcategory.gemini_client",
    "generate_content",
)
SUBCATEGORY_INTERFACE = "app_review_intelligence.components.subcategory.interface"
SUBCATEGORY_PACKAGE = "app_review_intelligence.components.subcategory"

FAKE_MODES = {
    "valid",           # every batch ID in one subcategory
    "unassigned",      # last batch ID listed in unassigned_review_ids
    "malformed",       # response is not JSON
    "invented_id",     # response adds an ID that was never sent
    "duplicate_id",    # first ID appears in two subcategories
    "missing_id",      # last ID is missing from the response
    "wrong_category",  # major_category does not match the request
}


def _collect_text(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _collect_text(item)]
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in _collect_text(item)]
    return [str(value)]


def _ids_in(text, known_ids):
    return [
        review_id
        for review_id in known_ids
        if re.search(rf"(?<![\w-]){re.escape(review_id)}(?![\w-])", text)
    ]


def _fake_response(state, batch_ids, mode):
    if mode == "malformed":
        return "Sure! Here are the groups: {not valid json"

    records = state["records"]
    category = records[batch_ids[0]]["predicted_major_category"] if batch_ids else "Other"
    code = CATEGORY_CODES.get(category, "OTHR")
    ids = list(batch_ids)
    unassigned = []

    if mode == "unassigned" and len(ids) > 1:
        unassigned = [ids.pop()]
    if mode == "missing_id" and len(ids) > 1:
        ids.pop()
    if mode == "invented_id":
        ids.append("INVENTED_999")

    subcategories = [{
        "subcategory_id": f"{code}_001",
        "subcategory_name": state["subcategory_name"],
        "description": state["subcategory_description"],
        "review_ids": ids,
    }]
    if mode == "duplicate_id":
        subcategories.append({
            "subcategory_id": f"{code}_002",
            "subcategory_name": "Duplicate Group",
            "description": "A second group that repeats a review ID.",
            "review_ids": ids[:1],
        })

    return json.dumps({
        "major_category": "Something Else" if mode == "wrong_category" else category,
        "subcategories": subcategories,
        "unassigned_review_ids": unassigned,
    })


@pytest.fixture
def fake_gemini(monkeypatch):
    """
    Replace the Gemini request function with a deterministic fake.

    Usage:
        state = fake_gemini(records, mode="valid", bad_calls=None)
        result = discover_subcategories(records)
        state["calls"]  # list of prompt texts, one per request

    `mode` (see FAKE_MODES) applies to every request, or only to the first
    `bad_calls` requests when `bad_calls` is given; later requests are valid.
    """
    _import_or_skip(SUBCATEGORY_INTERFACE)
    seam_module = _import_or_skip(GEMINI_SEAM[0])
    original = getattr(seam_module, GEMINI_SEAM[1], None)
    if original is None:
        pytest.skip(f"{GEMINI_SEAM[1]} has not been implemented in {GEMINI_SEAM[0]}")

    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)

    state = {
        "records": {},
        "mode": "valid",
        "bad_calls": None,
        "calls": [],
        "subcategory_name": "Application Crashes",
        "subcategory_description": "Reviews describing unexpected application shutdowns.",
    }

    def fake_generate_content(*args, **kwargs):
        prompt = "\n".join(_collect_text(list(args) + list(kwargs.values())))
        state["calls"].append(prompt)
        mode = state["mode"]
        if state["bad_calls"] is not None and len(state["calls"]) > state["bad_calls"]:
            mode = "valid"
        return _fake_response(state, _ids_in(prompt, state["records"]), mode)

    # Patch the seam wherever it was imported by name inside the component.
    for name, module in list(sys.modules.items()):
        if module is not None and (name == SUBCATEGORY_PACKAGE or name.startswith(SUBCATEGORY_PACKAGE + ".")):
            for attribute, value in list(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, attribute, fake_generate_content)

    def configure(records, mode="valid", bad_calls=None):
        assert mode in FAKE_MODES, mode
        state["records"] = {record["review_id"]: record for record in records}
        state["mode"] = mode
        state["bad_calls"] = bad_calls
        state["calls"].clear()
        return state

    return configure
