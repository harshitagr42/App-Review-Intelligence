"""Public interface for preprocessing. No logic lives here.

Example:
    from app_review_intelligence.preprocessing.interface import preprocess_records

    records = [
        {"review_id": "R0001", "review_text": "  App  crashes\\non login ", "review_date": "2026-07-21"},
    ]
    cleaned = preprocess_records(records)
    # [{"review_id": "R0001", "review_text": "App crashes on login", "review_date": "2026-07-21"}]
"""

from app_review_intelligence.preprocessing.implementation import (
    clean_and_validate_records,
)


def preprocess_records(records):
    """
    Cleans and validates review records.

    Input:
        records: list of {"review_id", "review_text", "review_date"} dicts.
            review_date may be None (no date column in the input).

    Returns:
        Cleaned review records with the same three fields. Records with empty
        text, an empty/unparseable date, or the same text and date as an
        earlier record are dropped.

    Raises:
        TypeError / ValueError when records are malformed or review_ids repeat.
    """
    cleaned, _ = clean_and_validate_records(records)
    return cleaned


def preprocess_records_with_report(records):
    """
    Same as preprocess_records, but also returns a report dict:
    input_count, accepted_count, skipped_empty_text (ids),
    skipped_invalid_date ([{review_id, raw_date}]),
    duplicates_removed ([{review_id, duplicate_of_review_id}]),
    text_modified_count, warnings.
    """
    return clean_and_validate_records(records)
