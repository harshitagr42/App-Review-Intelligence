"""Response parser and validator for Gemini subcategory output.

Pure functions with no I/O.
"""

import copy
import json
import re


def parse_response(raw_text):
    """
    Parses and checks the schema of Gemini's JSON response text.

    Strips optional markdown code fences and verifies that the top-level
    structure is a dict containing required fields: major_category (str),
    subcategories (list of dicts with subcategory_name, description, review_ids),
    and unassigned_review_ids (list). Any model-provided subcategory_id is ignored.

    Args:
        raw_text: Raw response string from Gemini.

    Returns:
        A normalized dictionary with keys 'major_category', 'subcategories',
        and 'unassigned_review_ids'.

    Raises:
        ValueError: If the raw text is not valid JSON or does not match the schema.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Response must be a string")

    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Failed to parse response as JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Response JSON must be an object at top level")

    if "major_category" not in data or not isinstance(data["major_category"], str):
        raise ValueError("Response missing string 'major_category'")

    if "subcategories" not in data or not isinstance(data["subcategories"], list):
        raise ValueError("Response missing list 'subcategories'")

    parsed_subcategories = []
    for index, sub in enumerate(data["subcategories"]):
        if not isinstance(sub, dict):
            raise ValueError(f"Subcategory item at index {index} is not an object")
        if "subcategory_name" not in sub or not isinstance(sub["subcategory_name"], str):
            raise ValueError(f"Subcategory at index {index} missing string 'subcategory_name'")
        if "description" not in sub or not isinstance(sub["description"], str):
            raise ValueError(f"Subcategory at index {index} missing string 'description'")
        if "review_ids" not in sub or not isinstance(sub["review_ids"], list):
            raise ValueError(f"Subcategory at index {index} missing list 'review_ids'")

        # Ensure all review_ids are strings
        cleaned_ids = [str(rid) for rid in sub["review_ids"]]
        parsed_subcategories.append({
            "subcategory_name": sub["subcategory_name"].strip(),
            "description": sub["description"].strip(),
            "review_ids": cleaned_ids,
        })

    unassigned = data.get("unassigned_review_ids", [])
    if not isinstance(unassigned, list):
        raise ValueError("'unassigned_review_ids' must be a list if present")
    cleaned_unassigned = [str(rid) for rid in unassigned]

    return {
        "major_category": data["major_category"].strip(),
        "subcategories": parsed_subcategories,
        "unassigned_review_ids": cleaned_unassigned,
    }


def validate_against_batch(parsed, batch_ids, expected_category):
    """
    Validates parsed model output against the requested batch.

    Checks for:
    - wrong_category: major_category differs from expected_category
    - invented_ids: response includes IDs not in batch_ids
    - duplicate_ids: an ID appears multiple times in subcategories or unassigned
    - missing_ids: an ID in batch_ids is not present in any subcategory or unassigned

    Empty review_ids lists inside a subcategory are ignored (dropped during processing).

    Args:
        parsed: Dict returned by parse_response.
        batch_ids: Collection of review ID strings expected in this batch.
        expected_category: String expected major category name.

    Returns:
        List of problem strings ('wrong_category', 'invented_ids',
        'duplicate_ids', 'missing_ids'), empty if valid.
    """
    problems = []
    if parsed.get("major_category") != expected_category:
        problems.append("wrong_category")

    batch_id_set = set(batch_ids)
    seen_ids = set()
    has_duplicates = False
    has_invented = False

    for sub in parsed.get("subcategories", []):
        for rid in sub.get("review_ids", []):
            if rid in seen_ids:
                has_duplicates = True
            seen_ids.add(rid)
            if rid not in batch_id_set:
                has_invented = True

    for rid in parsed.get("unassigned_review_ids", []):
        if rid in seen_ids:
            has_duplicates = True
        seen_ids.add(rid)
        if rid not in batch_id_set:
            has_invented = True

    if has_invented:
        problems.append("invented_ids")
    if has_duplicates:
        problems.append("duplicate_ids")

    missing = batch_id_set - seen_ids
    if missing:
        problems.append("missing_ids")

    return problems


def repair(parsed, batch_ids):
    """
    Repairs a parsed response after retries are exhausted.

    - Removes invented IDs.
    - Keeps the first occurrence of duplicate IDs.
    - Drops subcategories that have no remaining review IDs.
    - Moves missing batch IDs into unassigned_review_ids.

    Args:
        parsed: Dict returned by parse_response.
        batch_ids: Ordered sequence of review ID strings expected in the batch.

    Returns:
        Tuple of (repaired_dict, changes_list), where changes_list contains
        dicts of {'review_id': str, 'problem': str}.
    """
    batch_id_set = set(batch_ids)
    seen_ids = set()
    changes = []

    repaired_subcategories = []
    for sub in parsed.get("subcategories", []):
        repaired_rids = []
        for rid in sub.get("review_ids", []):
            if rid not in batch_id_set:
                changes.append({"review_id": rid, "problem": "invented_id"})
            elif rid in seen_ids:
                changes.append({"review_id": rid, "problem": "duplicate_id"})
            else:
                seen_ids.add(rid)
                repaired_rids.append(rid)

        if repaired_rids:
            repaired_subcategories.append({
                "subcategory_name": sub["subcategory_name"],
                "description": sub["description"],
                "review_ids": repaired_rids,
            })

    repaired_unassigned = []
    for rid in parsed.get("unassigned_review_ids", []):
        if rid not in batch_id_set:
            changes.append({"review_id": rid, "problem": "invented_id"})
        elif rid in seen_ids:
            changes.append({"review_id": rid, "problem": "duplicate_id"})
        else:
            seen_ids.add(rid)
            repaired_unassigned.append(rid)

    # Move any missing batch IDs into unassigned (in input order)
    for rid in batch_ids:
        if rid not in seen_ids:
            repaired_unassigned.append(rid)
            seen_ids.add(rid)
            changes.append({"review_id": rid, "problem": "missing_id"})

    repaired = {
        "major_category": parsed.get("major_category", ""),
        "subcategories": repaired_subcategories,
        "unassigned_review_ids": repaired_unassigned,
    }
    return repaired, changes
