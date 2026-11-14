
from conftest import get_component_function


def test_preprocessing_returns_list(sample_records):
    preprocess_records = get_component_function(
        "app_review_intelligence.preprocessing.interface",
        "preprocess_records",
    )

    result = preprocess_records(sample_records)

    assert isinstance(result, list)
    assert len(result) == len(sample_records)


def test_preprocessing_preserves_review_ids(sample_records):
    preprocess_records = get_component_function(
        "app_review_intelligence.preprocessing.interface",
        "preprocess_records",
    )

    result = preprocess_records(sample_records)

    original_ids = {r["review_id"] for r in sample_records}
    processed_ids = {r["review_id"] for r in result}

    assert original_ids == processed_ids


def test_preprocessing_does_not_remove_review_text(sample_records):
    preprocess_records = get_component_function(
        "app_review_intelligence.preprocessing.interface",
        "preprocess_records",
    )

    result = preprocess_records(sample_records)

    for record in result:
        assert "review_text" in record
        assert isinstance(record["review_text"], str)
        assert record["review_text"].strip() != ""


# ---------------------------------------------------------------------------
# Person 1: additional preprocessing tests
# ---------------------------------------------------------------------------

import pytest

PRE = ("app_review_intelligence.preprocessing.interface", "preprocess_records")


def _rec(review_id, text, date="2026-07-21"):
    return {"review_id": review_id, "review_text": text, "review_date": date}


def test_text_is_cleaned_without_changing_meaning():
    preprocess_records = get_component_function(*PRE)

    result = preprocess_records([_rec("R1", "  App\tcrashes\n\non  login!! 😡​ ")])

    assert result[0]["review_text"] == "App crashes on login!! 😡"


def test_empty_and_whitespace_text_is_dropped():
    preprocess_records = get_component_function(*PRE)

    result = preprocess_records([_rec("R1", ""), _rec("R2", "   \n"), _rec("R3", None), _rec("R4", "ok")])

    assert [r["review_id"] for r in result] == ["R4"]


def test_empty_or_invalid_date_is_dropped_but_none_is_kept():
    preprocess_records = get_component_function(*PRE)

    result = preprocess_records([
        _rec("R1", "a", date=""),
        _rec("R2", "b", date="31/31/2026"),
        _rec("R3", "c", date=None),
        _rec("R4", "d", date="2026-07-21 00:00:00"),
        _rec("R5", "e", date="2026/07/22"),
    ])

    assert [(r["review_id"], r["review_date"]) for r in result] == [
        ("R3", None), ("R4", "2026-07-21"), ("R5", "2026-07-22"),
    ]


def test_duplicates_need_same_text_and_same_date():
    preprocess_records = get_component_function(*PRE)

    result = preprocess_records([
        _rec("R1", "Slow app"),
        _rec("R2", "slow  APP "),
        _rec("R3", "Slow app", date="2026-08-01"),
    ])

    assert [r["review_id"] for r in result] == ["R1", "R3"]


def test_report_lists_every_dropped_record():
    report_fn = get_component_function(
        "app_review_intelligence.preprocessing.interface", "preprocess_records_with_report"
    )

    cleaned, report = report_fn([_rec("R1", "x"), _rec("R2", ""), _rec("R3", "y", ""), _rec("R4", "X")])

    assert [r["review_id"] for r in cleaned] == ["R1"]
    assert report["skipped_empty_text"] == ["R2"]
    assert report["skipped_invalid_date"] == [{"review_id": "R3", "raw_date": ""}]
    assert report["duplicates_removed"] == [{"review_id": "R4", "duplicate_of_review_id": "R1"}]


def test_preprocessing_is_idempotent(sample_records):
    preprocess_records = get_component_function(*PRE)

    once = preprocess_records(sample_records)

    assert preprocess_records(once) == once


def test_duplicate_review_ids_are_rejected():
    preprocess_records = get_component_function(*PRE)

    with pytest.raises(ValueError):
        preprocess_records([_rec("R1", "a"), _rec("R1", "b")])


def test_negation_and_punctuation_are_preserved():
    preprocess_records = get_component_function(*PRE)

    text = "I don't think it's NOT worth it... can't log in?!"
    result = preprocess_records([_rec("R1", text)])

    assert result[0]["review_text"] == text


def test_input_records_are_not_modified():
    preprocess_records = get_component_function(*PRE)
    records = [_rec("R1", "  spaced   text "), _rec("R2", "")]
    before = [dict(record) for record in records]

    preprocess_records(records)

    assert records == before
