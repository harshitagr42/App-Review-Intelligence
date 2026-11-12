"""Review-record cleaning and validation (Person 1).

Works on model-facing records only: ``{"review_id", "review_text", "review_date"}``.
Gold labels never pass through this module.

Rules applied, in input order:

1. ``review_text`` is cleaned without changing its meaning: Unicode NFC,
   zero-width characters removed, control characters turned into spaces,
   whitespace runs collapsed, ends trimmed. Case, punctuation and emoji are kept.
2. A record whose cleaned text is empty is dropped.
3. ``review_date``:
   - ``None`` means the input had no date column; the record is kept with a null date.
   - Any other value must parse as a calendar date and is normalised to ``YYYY-MM-DD``.
     An empty or unparseable value drops the record.
4. Exact duplicates are dropped, keeping the first occurrence. Two records are
   duplicates when their cleaned text matches case-insensitively *and* their
   normalised dates are equal. The same text on a different date is kept.

Nothing is written back to any input file; dropping only affects the records
handed to the prediction components.
"""

import logging
import re
import unicodedata
from datetime import date, datetime

logger = logging.getLogger(__name__)

_REMOVED_ZERO_WIDTH = dict.fromkeys(map(ord, "​⁠﻿"), None)
_WHITESPACE_RUN = re.compile(r"\s+")
_EXTRA_DATE_FORMATS = ("%Y/%m/%d", "%Y.%m.%d")


def clean_review_text(text):
    """Return the cleaned review text, or ``""`` when nothing meaningful is left."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_REMOVED_ZERO_WIDTH)
    # Control characters (category Cc: tabs, newlines, NUL, ...) become spaces.
    # Format characters such as the emoji zero-width joiner are kept.
    text = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def normalize_review_date(value):
    """Return ``YYYY-MM-DD`` for a parseable date, otherwise ``None``."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        # Handles "2026-07-21", "2026-07-21 00:00:00" (Excel cells) and ISO timestamps.
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for date_format in _EXTRA_DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def _validate_input(records):
    if not isinstance(records, list):
        raise TypeError("records must be a list of dicts")
    seen_ids = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"record at position {position} is not a dict")
        review_id = record.get("review_id")
        if review_id is None or str(review_id).strip() == "":
            raise ValueError(f"record at position {position} has no review_id")
        if review_id in seen_ids:
            raise ValueError(f"duplicate review_id {review_id!r} in records")
        seen_ids.add(review_id)


def clean_and_validate_records(records):
    """Clean, validate and de-duplicate records.

    Returns:
        (cleaned_records, report). ``report`` lists every dropped ``review_id``
        and why, so the pipeline can put the counts in ``run_metadata.json``.
    """
    _validate_input(records)

    cleaned = []
    skipped_empty_text = []
    skipped_invalid_date = []
    duplicates_removed = []
    text_modified_count = 0
    first_id_by_key = {}

    for record in records:
        review_id = record["review_id"]
        raw_text = record.get("review_text")
        text = clean_review_text(raw_text)
        if not text:
            skipped_empty_text.append(review_id)
            continue
        if text != raw_text:
            text_modified_count += 1

        raw_date = record.get("review_date")
        if raw_date is None:
            review_date = None
        else:
            review_date = normalize_review_date(raw_date)
            if review_date is None:
                skipped_invalid_date.append(
                    {"review_id": review_id, "raw_date": str(raw_date)}
                )
                continue

        key = (text.casefold(), review_date)
        if key in first_id_by_key:
            duplicates_removed.append(
                {"review_id": review_id, "duplicate_of_review_id": first_id_by_key[key]}
            )
            continue
        first_id_by_key[key] = review_id

        cleaned.append(
            {"review_id": review_id, "review_text": text, "review_date": review_date}
        )

    warnings = []
    if skipped_empty_text:
        warnings.append(f"{len(skipped_empty_text)} review(s) dropped: empty review text")
    if skipped_invalid_date:
        warnings.append(
            f"{len(skipped_invalid_date)} review(s) dropped: empty or unparseable date"
        )
    if duplicates_removed:
        warnings.append(
            f"{len(duplicates_removed)} review(s) dropped: duplicate text and date"
        )
    for message in warnings:
        logger.warning(message)

    report = {
        "input_count": len(records),
        "accepted_count": len(cleaned),
        "skipped_empty_text": skipped_empty_text,
        "skipped_invalid_date": skipped_invalid_date,
        "duplicates_removed": duplicates_removed,
        "text_modified_count": text_modified_count,
        "warnings": warnings,
    }
    return cleaned, report
