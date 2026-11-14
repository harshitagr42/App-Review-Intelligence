"""Public interface for subcategory discovery. No logic lives here.

Example:
    from app_review_intelligence.components.subcategory.interface import (
        discover_subcategories,
    )

    records = [
        {
            "review_id": "R0001",
            "review_text": "The app crashes whenever I try to log in.",
            "predicted_major_category": "Performance and Stability",
        }
    ]
    result = discover_subcategories(records)
    # {
    #     "subcategories": [
    #         {
    #             "subcategory_id": "PERF_001",
    #             "subcategory_name": "Application Crashes",
    #             "description": "Reviews describing unexpected application shutdowns.",
    #             "major_category": "Performance and Stability",
    #         }
    #     ],
    #     "assignments": [
    #         {"review_id": "R0001", "subcategory_id": "PERF_001"}
    #     ],
    # }
"""

from app_review_intelligence.components.subcategory.implementation import (
    run_discovery,
)


def discover_subcategories(records_with_major_categories, config=None):
    """
    Discovers subcategories from reviews grouped by predicted major category.

    Input:
        records_with_major_categories: list of dicts with review_id, review_text,
            and predicted_major_category.
        config: optional dict of configuration overrides (e.g. batch_size,
            max_retries, model_id).

    Returns:
        dict with exactly two keys:
            "subcategories": list of {subcategory_id, subcategory_name, description,
                major_category} dicts.
            "assignments": list of {review_id, subcategory_id} dicts in the same order
                as input. Unassigned reviews have subcategory_id "UNASSIGNED".

    Raises:
        ValueError: if input is invalid (bad fields, bad category name, duplicate IDs).
        RuntimeError: if API calls fail or responses cannot be parsed/repaired.
    """
    result, _ = run_discovery(records_with_major_categories, config=config)
    return result


def discover_subcategories_with_report(records_with_major_categories, config=None):
    """
    Same as discover_subcategories, but also returns a report dict.

    Returns:
        (result, report) tuple where report contains execution counts and diagnostics:
        batches, api_calls, retries, unassigned_review_ids, repaired, merges,
        model_id, and per-category review counts.
    """
    return run_discovery(records_with_major_categories, config=config)
