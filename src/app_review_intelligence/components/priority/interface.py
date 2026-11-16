"""Public interface for priority scoring. No logic lives here.

Example:
    from app_review_intelligence.components.priority.interface import calculate_priorities

    calculate_priorities(subcategory_result, sentiment_results, records=model_records)
    # [{"subcategory_id": "PERF_001", "priority_score": 72.4, "priority_level": "High", ...}]
"""

from app_review_intelligence.components.priority.implementation import (
    calculate_priorities as _calculate_priorities,
)


def calculate_priorities(subcategories, sentiment_results, records=None, config=None):
    """
    Calculates a V/N/S/T priority for each discovered subcategory.

    Input:
        subcategories: the full subcategory result
            {"subcategories": [{subcategory_id, subcategory_name, description, major_category}],
             "assignments": [{review_id, subcategory_id}]}
        sentiment_results: [{"review_id", "sentiment", "confidence"}]
        records: optional model-facing records [{"review_id", "review_text", "review_date"}].
            Dates enable trend; without them trend is unavailable.
            Severity comes from each subcategory's major category (settings.yaml).
        config: optional `priority` settings section (default: config/settings.yaml).

    Returns:
        One result per subcategory (UNASSIGNED excluded), sorted by score desc:
        {subcategory_id, subcategory_name, review_count, negative_proportion,
         priority_score, priority_level, explanation, components}.
    """
    return _calculate_priorities(subcategories, sentiment_results, records=records, config=config)
