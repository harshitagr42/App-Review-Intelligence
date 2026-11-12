
import pandas as pd
import pytest

from conftest import get_component_function


def test_load_and_prepare_returns_three_outputs(tmp_path):
    load_and_prepare = get_component_function(
        "app_review_intelligence.adapters.input.interface",
        "load_and_prepare",
    )

    file_path = tmp_path / "reviews.csv"

    df = pd.DataFrame([
        {
            "ID": "R0001",
            "Review": "The application crashes.",
            "Date": "2026-07-21",
            "Major Class": "Performance and Stability",
            "Minor Class": "App Crashes",
            "Sentiment": "Negative",
        }
    ])

    df.to_csv(file_path, index=False)

    model_records, evaluation_labels, metadata = load_and_prepare(
        [str(file_path)]
    )

    assert isinstance(model_records, list)
    assert isinstance(evaluation_labels, list)
    assert isinstance(metadata, dict)

    assert len(model_records) == 1
    assert len(evaluation_labels) == 1


def test_ground_truth_is_not_in_model_records(tmp_path):
    load_and_prepare = get_component_function(
        "app_review_intelligence.adapters.input.interface",
        "load_and_prepare",
    )

    file_path = tmp_path / "reviews.csv"

    pd.DataFrame([{
        "ID": "R0001",
        "Review": "Very slow application.",
        "Date": "2026-07-21",
        "Major Class": "Performance and Stability",
        "Minor Class": "Slow Performance",
        "Sentiment": "Negative",
    }]).to_csv(file_path, index=False)

    model_records, evaluation_labels, _ = load_and_prepare(
        [str(file_path)]
    )

    assert "review_text" in model_records[0]
    assert "review_id" in model_records[0]

    assert "gold_major_category" not in model_records[0]
    assert "gold_minor_category" not in model_records[0]
    assert "gold_sentiment" not in model_records[0]

    assert "gold_major_category" in evaluation_labels[0]
    assert "gold_minor_category" in evaluation_labels[0]
    assert "gold_sentiment" in evaluation_labels[0]


def test_duplicate_review_ids_are_rejected_or_corrected(tmp_path):
    load_and_prepare = get_component_function(
        "app_review_intelligence.adapters.input.interface",
        "load_and_prepare",
    )

    file_path = tmp_path / "reviews.csv"

    pd.DataFrame([
        {
            "ID": "R0001",
            "Review": "First review",
            "Date": "2026-07-21",
            "Major Class": "Usability",
            "Minor Class": "Navigation",
            "Sentiment": "Positive",
        },
        {
            "ID": "R0001",
            "Review": "Second review",
            "Date": "2026-07-22",
            "Major Class": "Usability",
            "Minor Class": "Navigation",
            "Sentiment": "Positive",
        },
    ]).to_csv(file_path, index=False)

    try:
        records, _, _ = load_and_prepare([str(file_path)])
    except ValueError:
        return

    ids = [record["review_id"] for record in records]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Person 1: additional input-adapter tests
# ---------------------------------------------------------------------------

from pathlib import Path

LOAD = ("app_review_intelligence.adapters.input.interface", "load_and_prepare")


def _row(review_id, text, date="2026-07-21", major="Usability",
         minor="Navigation", sentiment="positive"):
    return {"ID": review_id, "Review": text, "Date": date, "Major Class": major,
            "Minor Class": minor, "Sentiment": sentiment}


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_xlsx_input_loads(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = tmp_path / "reviews.xlsx"
    pd.DataFrame([_row("R1", "Great app"), _row("R2", "Too slow")]).to_excel(path, index=False)

    records, labels, metadata = load_and_prepare([str(path)])

    assert [r["review_id"] for r in records] == ["R1", "R2"]
    assert records[0]["review_date"] == "2026-07-21"
    assert metadata["input_files"][0]["file_type"] == "xlsx"
    assert len(labels) == 2


def test_two_files_are_combined_with_unique_ids(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    first = _write_csv(tmp_path / "a.csv", [_row("R1", "First file review")])
    second = _write_csv(tmp_path / "b.csv", [_row("R1", "Second file review")])

    records, labels, metadata = load_and_prepare([first, second])

    ids = [r["review_id"] for r in records]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert {label["review_id"] for label in labels} == set(ids)
    assert metadata["id_corrections"][0]["source_review_id"] == "R1"


def test_column_mapping(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = tmp_path / "custom.csv"
    pd.DataFrame([{"rid": "X1", "body": "Nice", "when": "2026-01-02"}]).to_csv(path, index=False)

    records, labels, _ = load_and_prepare(
        [str(path)], id_column="rid", text_column="body", date_column="when"
    )

    assert records == [{"review_id": "X1", "review_text": "Nice", "review_date": "2026-01-02"}]
    assert labels == []


def test_missing_text_column_fails(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"ID": "R1", "Text": "wrong column"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Review"):
        load_and_prepare([str(path)])


def test_empty_file_fails(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    header_only = tmp_path / "header.csv"
    header_only.write_text("ID,Review,Date\n")

    with pytest.raises(ValueError):
        load_and_prepare([str(empty)])
    with pytest.raises(ValueError):
        load_and_prepare([str(header_only)])


def test_missing_file_and_empty_list_fail(tmp_path):
    load_and_prepare = get_component_function(*LOAD)

    with pytest.raises(FileNotFoundError):
        load_and_prepare([str(tmp_path / "nope.csv")])
    with pytest.raises(ValueError):
        load_and_prepare([])


def test_unsupported_extension_fails(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = tmp_path / "reviews.txt"
    path.write_text("ID,Review\nR1,hello\n")

    with pytest.raises(ValueError, match="Unsupported"):
        load_and_prepare([str(path)])


def test_invalid_rows_and_duplicates_are_dropped(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = _write_csv(tmp_path / "mixed.csv", [
        _row("1", "Keeps crashing on login"),
        _row("2", ""),                                    # empty review
        _row("3", "Has text but no date", date=""),       # empty date
        _row("4", "", date=""),                           # both empty
        _row("5", "Bad date", date="not-a-date"),         # unparseable date
        _row("6", "  keeps   CRASHING on login "),        # duplicate of 1
        _row("7", "Keeps crashing on login", date="2026-08-01"),  # same text, other date
    ])

    records, labels, metadata = load_and_prepare([path])

    assert [r["review_id"] for r in records] == ["1", "7"]
    assert [label["review_id"] for label in labels] == ["1", "7"]
    assert metadata["rows_loaded"] == 7
    assert metadata["rows_accepted"] == 2
    assert metadata["rows_skipped_empty"] == 2
    assert metadata["rows_skipped_invalid_date"] == 2
    assert metadata["duplicate_review_count"] == 1
    assert metadata["preprocessing"]["duplicates_removed"] == [
        {"review_id": "6", "duplicate_of_review_id": "1"}
    ]


def test_no_date_column_keeps_rows_with_null_date(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = tmp_path / "nodate.csv"
    pd.DataFrame([{"Review": "Good"}, {"Review": "Bad"}]).to_csv(path, index=False)

    records, _, metadata = load_and_prepare([str(path)])

    assert [r["review_date"] for r in records] == [None, None]
    assert [r["review_id"] for r in records] == ["R0001", "R0002"]
    assert any("Date" in w for w in metadata["warnings"])


def test_sentiment_labels_are_normalised(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = _write_csv(tmp_path / "s.csv", [
        _row("1", "a", sentiment="positive"),
        _row("2", "b", sentiment="NEGATIVE"),
        _row("3", "c", sentiment=" neutral "),
    ])

    _, labels, _ = load_and_prepare([path])

    assert [label["gold_sentiment"] for label in labels] == ["Positive", "Negative", "Neutral"]


def test_model_records_have_only_model_fields(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = _write_csv(tmp_path / "r.csv", [_row("1", "text")])

    records, _, _ = load_and_prepare([path])

    assert set(records[0]) == {"review_id", "review_text", "review_date"}


def test_all_rows_invalid_fails(tmp_path):
    load_and_prepare = get_component_function(*LOAD)
    path = _write_csv(tmp_path / "bad.csv", [_row("1", ""), _row("2", "x", date="")])

    with pytest.raises(ValueError, match="No valid reviews"):
        load_and_prepare([path])


COMBINED = Path(__file__).resolve().parents[1] / "dataset" / "processed" / "combined_reviews_output.csv"


@pytest.mark.skipif(not COMBINED.exists(), reason="local dataset not available")
def test_team_combined_dataset_is_cleaned():
    load_and_prepare = get_component_function(*LOAD)

    records, labels, metadata = load_and_prepare([str(COMBINED)])

    assert metadata["rows_loaded"] == 2500
    assert metadata["rows_accepted"] == 1997
    assert metadata["rows_skipped_empty"] == 15
    assert metadata["rows_skipped_invalid_date"] == 14
    assert metadata["duplicate_review_count"] == 474
    assert all(r["review_text"] and r["review_date"] for r in records)
    keys = {(r["review_text"].casefold(), r["review_date"]) for r in records}
    assert len(keys) == len(records)
    assert [label["review_id"] for label in labels] == [r["review_id"] for r in records]
