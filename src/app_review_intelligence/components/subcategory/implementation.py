"""Implementation of subcategory discovery for app reviews.

Groups reviews by predicted major category, queries Gemini in batches,
validates untrusted responses with bounded retries, repairs semantic defects,
and consolidates discovered subcategories.
"""

import copy
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path

import yaml

from app_review_intelligence.components.subcategory.consolidator import (
    UNASSIGNED,
    consolidate_subcategories,
)
from app_review_intelligence.components.subcategory.gemini_client import (
    DEFAULT_API_KEY_ENV_VAR,
    check_api_key,
    generate_content,
    is_transient_error,
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

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY_CODES = {
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


def _sleep(seconds):
    """
    Sleeps for the given duration. Module-level seam for testing.

    Args:
        seconds: Float seconds to sleep.
    """
    if seconds > 0:
        time.sleep(seconds)


def load_category_codes():
    """
    Loads category code mappings from config/major_categories.yaml or defaults.

    Returns:
        Dict mapping major category name to 3-4 letter code.
    """
    repo_root = Path(__file__).resolve().parents[4]
    categories_file = repo_root / "config" / "major_categories.yaml"
    if categories_file.is_file():
        try:
            with open(categories_file, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
                if isinstance(data, dict) and "categories" in data:
                    codes = {c["name"]: c["code"] for c in data["categories"]}
                    if codes:
                        return codes
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_CATEGORY_CODES)


def load_settings(config=None):
    """
    Loads discovery settings from config/settings.yaml and merges overrides.

    Args:
        config: Optional dict of override settings.

    Returns:
        Flat dictionary of configuration settings.
    """
    repo_root = Path(__file__).resolve().parents[4]
    settings_file = repo_root / "config" / "settings.yaml"

    raw = {}
    if settings_file.is_file():
        try:
            with open(settings_file, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
                if isinstance(loaded, dict):
                    raw = loaded
        except Exception:
            raw = {}

    llm_section = (
        raw.get("models", {}).get("llm", {})
        if isinstance(raw.get("models"), dict)
        else {}
    )
    embedding_section = (
        raw.get("models", {}).get("embedding", {})
        if isinstance(raw.get("models"), dict)
        else {}
    )

    settings = {
        "model_id": llm_section.get("model_id", "gemini-3.6-flash"),
        "api_key_env_var": llm_section.get("api_key_env_var", "GEMINI_API_KEY"),
        "batch_size": llm_section.get("batch_size", raw.get("llm_batch_size", 40)),
        "max_retries": llm_section.get("max_retries", 3),
        "merge_similarity": raw.get("llm_cluster_merge_similarity", 0.82),
        "embedding_model_id": embedding_section.get(
            "model_id", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "random_seed": raw.get("random_seed", 42),
        "min_request_interval_seconds": llm_section.get(
            "min_request_interval_seconds", 0
        ),
    }

    if config and isinstance(config, dict):
        for key, value in config.items():
            if value is not None:
                settings[key] = value

    return settings


def _format_batch_ids_for_error(batch_ids):
    """Formats batch IDs for concise error logging without exposing review text."""
    if len(batch_ids) <= 10:
        return ", ".join(batch_ids)
    return ", ".join(batch_ids[:10]) + f" (+{len(batch_ids) - 10} more)"


def run_discovery(records, config=None):
    """
    Orchestrates subcategory discovery across review records.

    Steps:
    1. Load settings and category taxonomy.
    2. Validate API key environment variable BEFORE making network requests.
    3. Validate input records and drop all extra fields.
    4. Group records by predicted major category.
    5. Chunk each group into category-homogeneous batches.
    6. Run bounded retry loop with exponential backoff on transient errors.
    7. Repair semantic problems when retries are exhausted.
    8. Consolidate subcategories and assign final IDs.
    9. Assert invariants before returning.

    Args:
        records: List of dicts with review_id, review_text, predicted_major_category.
        config: Optional configuration override dictionary.

    Returns:
        Tuple of (result_dict, report_dict).
        result_dict contains 'subcategories' and 'assignments'.
        report_dict contains counts and diagnostic metadata.

    Raises:
        ValueError: If input structure, category names, or review IDs are invalid.
        RuntimeError: If API requests fail or unrecoverable response errors persist.
    """
    # 1. Load settings
    settings = load_settings(config)
    category_codes = load_category_codes()

    # 2. Check API key BEFORE doing anything else
    api_key_var = settings.get("api_key_env_var", DEFAULT_API_KEY_ENV_VAR)
    os.environ["GEMINI_API_KEY_ENV_VAR"] = api_key_var
    check_api_key(api_key_var)

    # 3. Validate input
    if not isinstance(records, list):
        raise ValueError("records must be a list of dicts")

    if not records:
        empty_report = {
            "batches": 0,
            "api_calls": 0,
            "retries": 0,
            "unassigned_review_ids": [],
            "repaired": [],
            "merges": 0,
            "model_id": settings["model_id"],
            "categories": {},
        }
        return {"subcategories": [], "assignments": []}, empty_report

    sanitized_records = []
    seen_ids = set()
    invalid_categories = set()

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Record at index {idx} is not a dict")

        rid = item.get("review_id")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"Record at index {idx} has invalid or missing 'review_id'")
        rid = rid.strip()

        if rid in seen_ids:
            raise ValueError(f"Duplicate review_id found: '{rid}'")
        seen_ids.add(rid)

        text = item.get("review_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record '{rid}' has empty or non-string 'review_text'")

        category = item.get("predicted_major_category")
        if not isinstance(category, str) or category not in category_codes:
            invalid_categories.add(str(category))

        # Keep strictly only the three required fields; extra fields dropped
        sanitized_records.append({
            "review_id": rid,
            "review_text": text,
            "predicted_major_category": category,
        })

    if invalid_categories:
        bad_cats = sorted(invalid_categories)
        raise ValueError(
            f"Unknown predicted_major_category values found: {bad_cats}. "
            f"Must be one of {sorted(category_codes.keys())}."
        )

    # 4. Group by predicted major category (sorted order of categories)
    grouped_by_category = {cat: [] for cat in sorted(category_codes.keys())}
    category_counts = {}
    for r in sanitized_records:
        cat = r["predicted_major_category"]
        grouped_by_category[cat].append(r)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Load prompts
    system_prompt, user_template = load_prompts()

    # 5. Process batches per major category
    batch_size = max(1, int(settings.get("batch_size", 40)))
    max_retries = max(0, int(settings.get("max_retries", 3)))
    min_interval = float(settings.get("min_request_interval_seconds", 0))
    random_seed = int(settings.get("random_seed", 42))

    total_batches = 0
    total_api_calls = 0
    total_retries = 0
    all_repaired = []
    unassigned_ids = set()

    raw_subcategories_by_category = {cat: [] for cat in grouped_by_category}

    for cat in sorted(grouped_by_category.keys()):
        cat_records = grouped_by_category[cat]
        if not cat_records:
            continue

        # Chunk into batches
        batches = [
            cat_records[i : i + batch_size]
            for i in range(0, len(cat_records), batch_size)
        ]

        for batch_index, batch in enumerate(batches):
            total_batches += 1
            batch_ids = [r["review_id"] for r in batch]
            logger.info(
                "Processing category '%s' batch %d/%d (%d reviews)",
                cat,
                batch_index + 1,
                len(batches),
                len(batch),
            )

            # 6. Retry loop per batch
            max_attempts = 1 + max_retries
            attempt = 0
            transient_failures = 0
            retry_hint = ""
            batch_parsed = None
            last_failure_type = None
            last_problem_desc = ""

            while attempt < max_attempts:
                attempt += 1
                if total_api_calls > 0 and min_interval > 0:
                    _sleep(min_interval)

                temperature = 0.0 if attempt == 1 else 0.3
                prompt_content = build_user_prompt(user_template, cat, batch)
                if retry_hint:
                    prompt_content += retry_hint

                total_api_calls += 1
                if attempt > 1:
                    total_retries += 1

                try:
                    raw_response = generate_content(
                        system_prompt,
                        prompt_content,
                        temperature=temperature,
                        model_id=settings.get("model_id"),
                        seed=random_seed,
                    )
                except Exception as exc:
                    if is_transient_error(exc):
                        transient_failures += 1
                        last_failure_type = "api_error"
                        last_problem_desc = f"Transient API error: {exc}"
                        logger.warning(
                            "Transient API error on attempt %d for category '%s': %s",
                            attempt,
                            cat,
                            exc,
                        )
                        if attempt < max_attempts:
                            backoff = min(60.0, (2.0 ** (transient_failures - 1)) * 2.0)
                            jitter = random.uniform(0.0, 0.5)
                            _sleep(backoff + jitter)
                            continue
                    else:
                        logger.error(
                            "Permanent API error on attempt %d for category '%s': %s",
                            attempt,
                            cat,
                            exc,
                        )
                        raise

                # Parse response
                try:
                    parsed = parse_response(raw_response)
                except ValueError as parse_err:
                    last_failure_type = "bad_json"
                    last_problem_desc = f"Malformed JSON: {parse_err}"
                    logger.warning(
                        "Bad JSON response on attempt %d for category '%s'",
                        attempt,
                        cat,
                    )
                    retry_hint = build_retry_hint(["malformed_json"])
                    continue

                # Validate against batch
                problems = validate_against_batch(parsed, batch_ids, cat)
                if not problems:
                    # Valid response
                    batch_parsed = parsed
                    last_failure_type = None
                    break

                # Response has problems
                logger.warning(
                    "Validation problems on attempt %d for category '%s': %s",
                    attempt,
                    cat,
                    problems,
                )
                if "wrong_category" in problems:
                    last_failure_type = "wrong_category"
                    last_problem_desc = "major_category mismatch"
                else:
                    last_failure_type = "semantic"
                    last_problem_desc = f"Validation problems: {problems}"

                batch_parsed = parsed
                retry_hint = build_retry_hint(problems)

            # When attempts are exhausted:
            if last_failure_type in ("api_error", "bad_json", "wrong_category") or batch_parsed is None:
                formatted_ids = _format_batch_ids_for_error(batch_ids)
                raise RuntimeError(
                    f"Subcategory discovery failed for major category '{cat}' on batch "
                    f"with review IDs [{formatted_ids}]. Reason: {last_problem_desc}"
                )

            # Check if semantic repair is needed
            final_problems = validate_against_batch(batch_parsed, batch_ids, cat)
            if final_problems:
                logger.warning(
                    "Exhausted retries with semantic defects for category '%s' batch: %s. Applying repair.",
                    cat,
                    final_problems,
                )
                repaired_batch, changes = repair(batch_parsed, batch_ids)
                all_repaired.extend(changes)
                batch_parsed = repaired_batch

            # Collect subcategories and unassigned IDs from this batch
            for sub in batch_parsed.get("subcategories", []):
                raw_subcategories_by_category[cat].append(sub)

            for uid in batch_parsed.get("unassigned_review_ids", []):
                unassigned_ids.add(uid)

    # 7. Consolidate per category and assign final IDs
    total_merges = 0
    final_subcategories_all = []
    id_to_subcategory = {}

    for cat in sorted(grouped_by_category.keys()):
        raw_subs = raw_subcategories_by_category[cat]
        cat_code = category_codes.get(cat, "OTHR")

        cat_final_subs, cat_mapping, cat_merges = consolidate_subcategories(
            major_category=cat,
            category_code=cat_code,
            raw_subcategories=raw_subs,
            merge_similarity=float(settings.get("merge_similarity", 0.82)),
            embedding_model_id=settings.get("embedding_model_id"),
            random_seed=random_seed,
        )

        total_merges += cat_merges
        final_subcategories_all.extend(cat_final_subs)
        id_to_subcategory.update(cat_mapping)

    # 8. Build assignments in original INPUT order
    assignments = []
    final_unassigned_list = []
    for record in sanitized_records:
        rid = record["review_id"]
        sub_id = id_to_subcategory.get(rid, UNASSIGNED)
        assignments.append({"review_id": rid, "subcategory_id": sub_id})
        if sub_id == UNASSIGNED:
            final_unassigned_list.append(rid)

    # Sort subcategories by major category name then subcategory_id
    final_subcategories_all.sort(
        key=lambda s: (s["major_category"], s["subcategory_id"])
    )

    # 9. Assert invariants
    input_ids = [r["review_id"] for r in sanitized_records]
    assigned_ids = [a["review_id"] for a in assignments]
    if assigned_ids != input_ids:
        raise RuntimeError("Invariant violation: assignments do not match input review IDs or order")

    valid_sub_ids = {s["subcategory_id"] for s in final_subcategories_all}
    for a in assignments:
        if a["subcategory_id"] != UNASSIGNED and a["subcategory_id"] not in valid_sub_ids:
            raise RuntimeError(
                f"Invariant violation: assignment references unknown subcategory_id '{a['subcategory_id']}'"
            )

    result = {
        "subcategories": final_subcategories_all,
        "assignments": assignments,
    }

    report = {
        "batches": total_batches,
        "api_calls": total_api_calls,
        "retries": total_retries,
        "unassigned_review_ids": sorted(final_unassigned_list),
        "repaired": all_repaired,
        "merges": total_merges,
        "model_id": settings["model_id"],
        "categories": category_counts,
    }

    return result, report
