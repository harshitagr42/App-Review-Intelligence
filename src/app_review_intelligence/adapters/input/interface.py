"""Public interface for the input adapter. No logic lives here.

Example:
    from app_review_intelligence.adapters.input.interface import load_and_prepare

    model_records, evaluation_labels, metadata = load_and_prepare(
        ["dataset/processed/combined_reviews_output.csv"]
    )
    model_records[0]      # {"review_id": "1", "review_text": "...", "review_date": "2026-08-15"}
    evaluation_labels[0]  # {"review_id": "1", "gold_major_category": "...",
                          #  "gold_minor_category": "...", "gold_sentiment": "Negative"}
    metadata["rows_loaded"], metadata["rows_accepted"]   # 2500, 1997
"""

from app_review_intelligence.adapters.input.implementation import (
    DEFAULT_DATE_COLUMN,
    DEFAULT_ID_COLUMN,
    DEFAULT_TEXT_COLUMN,
    load_input_files,
)


def load_and_prepare(
    input_paths,
    id_column=DEFAULT_ID_COLUMN,
    text_column=DEFAULT_TEXT_COLUMN,
    date_column=DEFAULT_DATE_COLUMN,
    sheet_name=None,
):
    """
    Reads and combines input files.

    Input:
        input_paths: list of CSV/XLSX paths (a single path is also accepted).
        id_column / text_column / date_column: source column names
            (defaults "ID", "Review", "Date"). Only the text column is required.
        sheet_name: worksheet for XLSX files (default: first sheet).

    Returns:
        model_records: Records used by prediction components:
            [{"review_id", "review_text", "review_date"}]. Already preprocessed.
        evaluation_labels: Ground-truth labels used only for evaluation:
            [{"review_id", "gold_major_category", "gold_minor_category", "gold_sentiment"}].
            Empty when the input has no label columns.
        dataset_metadata: Information about the loaded dataset (files, column
            mapping, row counts, dropped rows, ID corrections, warnings).

    Raises:
        ValueError: no paths, unsupported/empty file, missing text column,
            or no valid reviews left.
        FileNotFoundError: a path does not exist.
    """
    return load_input_files(
        input_paths,
        id_column=id_column,
        text_column=text_column,
        date_column=date_column,
        sheet_name=sheet_name,
    )
