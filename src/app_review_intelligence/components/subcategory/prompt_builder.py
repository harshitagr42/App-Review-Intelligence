"""Prompt loading and construction for subcategory discovery."""

import json
from pathlib import Path


def load_prompts(prompts_dir=None):
    """
    Loads system prompt and user prompt template files.

    Args:
        prompts_dir: Optional Path or str pointing to the directory containing
            subcategory_system.txt and subcategory_user_template.txt.
            Defaults to repo_root/prompts.

    Returns:
        Tuple of (system_text, user_template_text).

    Raises:
        FileNotFoundError: If a file is missing or empty.
    """
    if prompts_dir is None:
        repo_root = Path(__file__).resolve().parents[4]
        prompts_path = repo_root / "prompts"
    else:
        prompts_path = Path(prompts_dir)

    system_file = prompts_path / "subcategory_system.txt"
    user_file = prompts_path / "subcategory_user_template.txt"

    if not system_file.is_file():
        raise FileNotFoundError(f"System prompt file not found: {system_file}")
    if not user_file.is_file():
        raise FileNotFoundError(f"User prompt template file not found: {user_file}")

    system_text = system_file.read_text(encoding="utf-8").strip()
    user_text = user_file.read_text(encoding="utf-8").strip()

    if not system_text:
        raise FileNotFoundError(f"System prompt file is empty: {system_file}")
    if not user_text:
        raise FileNotFoundError(f"User prompt template file is empty: {user_file}")

    return system_text, user_text


def build_user_prompt(template, major_category, batch_records):
    """
    Populates user prompt template with major category and batch reviews JSON.

    Fills {{MAJOR_CATEGORY}} and {{REVIEWS_JSON}} using str.replace.
    The reviews block contains only review ID and review text.

    Args:
        template: Template string containing {{MAJOR_CATEGORY}} and {{REVIEWS_JSON}}.
        major_category: String name of the predicted major category.
        batch_records: List of dicts, each having 'review_id' and 'review_text'.

    Returns:
        Formatted user prompt string.
    """
    reviews_data = [
        {"id": record["review_id"], "text": record["review_text"]}
        for record in batch_records
    ]
    reviews_json = json.dumps(reviews_data, ensure_ascii=False, indent=2)

    prompt = template.replace("{{MAJOR_CATEGORY}}", major_category)
    prompt = prompt.replace("{{REVIEWS_JSON}}", reviews_json)
    return prompt


def build_retry_hint(problem_summary, batch_ids=None):
    """
    Builds a short generic sentence listing what was wrong using only valid batch IDs.

    Never echoes unknown or invented strings from an invalid response.

    Args:
        problem_summary: dict or collection describing the validation failure,
            e.g. {'missing_ids': ['R0001', 'R0002']} or list of problem names.
        batch_ids: Optional sequence of valid batch review IDs to restrict listed IDs to.

    Returns:
        A concise retry hint string.
    """
    allowed_set = set(str(rid) for rid in batch_ids) if batch_ids is not None else None

    if isinstance(problem_summary, dict):
        if allowed_set is None and "batch_ids" in problem_summary:
            allowed_set = set(str(rid) for rid in problem_summary["batch_ids"])

        hints = []
        if "missing_ids" in problem_summary and problem_summary["missing_ids"]:
            raw_missing = problem_summary["missing_ids"]
            if allowed_set is not None:
                clean_ids = sorted(str(rid) for rid in raw_missing if str(rid) in allowed_set)
            else:
                clean_ids = sorted(str(rid) for rid in raw_missing)
            if clean_ids:
                hints.append(f"These IDs were missing: {clean_ids}.")
            else:
                hints.append("Some provided review IDs were missing from the response.")
        if "duplicate_ids" in problem_summary and problem_summary["duplicate_ids"]:
            hints.append("Some IDs appeared multiple times across groups.")
        if "invented_ids" in problem_summary and problem_summary["invented_ids"]:
            hints.append("Unrecognized IDs were found; use only the provided review IDs.")
        if "wrong_category" in problem_summary and problem_summary["wrong_category"]:
            hints.append("The major_category field did not match the requested category.")
        if hints:
            return " " + " ".join(hints) + " Assign every provided ID exactly once."
    elif isinstance(problem_summary, (list, tuple, set)):
        hints = []
        if "missing_ids" in problem_summary:
            hints.append("Some provided review IDs were missing from the response.")
        if "duplicate_ids" in problem_summary:
            hints.append("Some IDs were assigned to more than one group.")
        if "invented_ids" in problem_summary:
            hints.append("Do not invent IDs; use only the provided IDs.")
        if "wrong_category" in problem_summary:
            hints.append("The major_category must match the requested category.")
        if hints:
            return " " + " ".join(hints) + " Assign every provided ID exactly once."
    return " Please return valid JSON matching the schema and assign every provided ID exactly once."
