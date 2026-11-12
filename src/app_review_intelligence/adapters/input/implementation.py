"""Input adapter (Person 1): read, combine and split CSV/XLSX review files.

Steps:
1. Validate the path list and read every file as text (CSV or XLSX).
2. Check the required review-text column; the ID and date columns are optional.
3. Give every row a review_id that is unique across all files. Source IDs are
   kept when unique; missing IDs become ``R0001``-style IDs; repeated IDs get a
   ``_2``, ``_3`` ... suffix. Every correction is recorded in the metadata.
4. Split gold labels (``Major Class``, ``Minor Class``, ``Sentiment``) off into
   separate evaluation records. Model records never contain them.
5. Run the preprocessing interface to clean the text and drop rows with an empty
   review, an empty/invalid date, or a duplicate text+date. Evaluation labels are
   filtered to the accepted review_ids, so both lists stay aligned.

Input files are only read, never modified.
"""

import logging
from pathlib import Path

import pandas as pd

from app_review_intelligence.preprocessing.interface import (
    preprocess_records_with_report,
)

logger = logging.getLogger(__name__)

DEFAULT_ID_COLUMN = "ID"
DEFAULT_TEXT_COLUMN = "Review"
DEFAULT_DATE_COLUMN = "Date"

# Source column -> evaluation field. Used only to build evaluation labels.
GOLD_LABEL_COLUMNS = {
    "Major Class": "gold_major_category",
    "Minor Class": "gold_minor_category",
    "Sentiment": "gold_sentiment",
}

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
_SENTIMENT_LABELS = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}


def _resolve_paths(input_paths):
    if input_paths is None:
        raise ValueError("No input files were given")
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]
    paths = [Path(p) for p in input_paths]
    if not paths:
        raise ValueError("No input files were given")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type {path.suffix!r} for {path.name}; "
                f"use one of {sorted(SUPPORTED_EXTENSIONS)}"
            )
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same input file was given more than once")
    return paths


def _read_table(path, sheet_name):
    """Read one file with every cell as a string; empty cells become ""."""
    try:
        if path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            except UnicodeDecodeError:
                logger.warning("%s is not UTF-8; reading it as cp1252", path.name)
                frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp1252")
            used_sheet = None
        else:
            used_sheet = 0 if sheet_name is None else sheet_name
            frame = pd.read_excel(
                path, sheet_name=used_sheet, dtype=str, keep_default_na=False, engine="openpyxl"
            )
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty: {path.name}") from None
    except ValueError as error:
        if path.suffix.lower() == ".xlsx" and "Worksheet" in str(error):
            raise ValueError(f"Sheet {sheet_name!r} not found in {path.name}") from None
        raise

    frame = frame.fillna("")
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError(f"Input file has no data rows: {path.name}")
    return frame, used_sheet


def _cell(row, column):
    value = row.get(column, "")
    return "" if value is None else str(value).strip()


def _normalize_sentiment(value):
    return _SENTIMENT_LABELS.get(value.lower(), value)


def _unique_suffix_id(base, taken):
    counter = 2
    while f"{base}_{counter}" in taken:
        counter += 1
    return f"{base}_{counter}"


def _next_generated_id(state, taken):
    while True:
        state["counter"] += 1
        candidate = f"R{state['counter']:04d}"
        if candidate not in taken:
            return candidate


def load_input_files(
    input_paths,
    id_column=DEFAULT_ID_COLUMN,
    text_column=DEFAULT_TEXT_COLUMN,
    date_column=DEFAULT_DATE_COLUMN,
    sheet_name=None,
):
    """Read, combine, split and preprocess input files.

    Returns:
        (model_records, evaluation_labels, dataset_metadata)
    """
    paths = _resolve_paths(input_paths)

    tables = []
    for path in paths:
        frame, used_sheet = _read_table(path, sheet_name)
        if text_column not in frame.columns:
            raise ValueError(
                f"Required review text column {text_column!r} is missing in {path.name}. "
                f"Available columns: {list(frame.columns)}"
            )
        tables.append((path, frame, used_sheet))

    # Every source ID across all files, so generated or suffixed IDs never clash with one.
    all_source_ids = set()
    for _, frame, _ in tables:
        if id_column in frame.columns:
            all_source_ids.update(v.strip() for v in frame[id_column] if v.strip())

    warnings = []
    input_files = []
    raw_records = []
    labels_by_id = {}
    record_sources = {}
    id_corrections = []
    gold_columns_found = set()
    taken_ids = set(all_source_ids)
    assigned_ids = set()
    generated_state = {"counter": 0}

    for path, frame, used_sheet in tables:
        has_id = id_column in frame.columns
        has_date = date_column in frame.columns
        file_gold = [c for c in GOLD_LABEL_COLUMNS if c in frame.columns]
        gold_columns_found.update(file_gold)
        if not has_id:
            warnings.append(f"{path.name}: no {id_column!r} column; review IDs were generated")
        if not has_date:
            warnings.append(f"{path.name}: no {date_column!r} column; review dates are null")

        input_files.append(
            {
                "path": str(path),
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "sheet_name": used_sheet,
                "row_count": int(len(frame)),
            }
        )

        for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
            source_id = _cell(row, id_column) if has_id else ""
            if source_id and source_id not in assigned_ids:
                review_id = source_id
            elif source_id:
                review_id = _unique_suffix_id(source_id, taken_ids)
                id_corrections.append(
                    {
                        "review_id": review_id,
                        "source_review_id": source_id,
                        "source_file": path.name,
                        "reason": "duplicate source ID",
                    }
                )
            else:
                review_id = _next_generated_id(generated_state, taken_ids)
                if has_id:
                    id_corrections.append(
                        {
                            "review_id": review_id,
                            "source_review_id": None,
                            "source_file": path.name,
                            "reason": "missing source ID",
                        }
                    )
            taken_ids.add(review_id)
            assigned_ids.add(review_id)

            raw_records.append(
                {
                    "review_id": review_id,
                    "review_text": row.get(text_column, ""),
                    # None = no date column at all; "" = empty cell (dropped in preprocessing).
                    "review_date": row.get(date_column, "") if has_date else None,
                }
            )
            record_sources[review_id] = {
                "source_file": path.name,
                "source_review_id": source_id or None,
                "source_row": row_number,
            }

            if file_gold:
                label = {"review_id": review_id}
                for column, field in GOLD_LABEL_COLUMNS.items():
                    value = _cell(row, column) if column in file_gold else ""
                    if value and column == "Sentiment":
                        value = _normalize_sentiment(value)
                    label[field] = value or None
                labels_by_id[review_id] = label

    if id_corrections:
        warnings.append(f"{len(id_corrections)} review ID(s) were generated or de-duplicated")

    model_records, report = preprocess_records_with_report(raw_records)
    if not model_records:
        raise ValueError("No valid reviews remain after preprocessing")

    accepted_ids = [record["review_id"] for record in model_records]
    evaluation_labels = [labels_by_id[i] for i in accepted_ids if i in labels_by_id]

    for message in warnings:
        logger.warning(message)

    dataset_metadata = {
        "input_files": input_files,
        "column_mapping": {"id": id_column, "text": text_column, "date": date_column},
        "gold_label_columns_found": [c for c in GOLD_LABEL_COLUMNS if c in gold_columns_found],
        "rows_loaded": len(raw_records),
        "rows_accepted": len(model_records),
        "rows_skipped_empty": len(report["skipped_empty_text"]),
        "rows_skipped_invalid_date": len(report["skipped_invalid_date"]),
        "duplicate_review_count": len(report["duplicates_removed"]),
        "evaluation_label_count": len(evaluation_labels),
        "id_corrections": id_corrections,
        "preprocessing": report,
        "record_sources": record_sources,
        "warnings": warnings + report["warnings"],
    }
    return model_records, evaluation_labels, dataset_metadata
