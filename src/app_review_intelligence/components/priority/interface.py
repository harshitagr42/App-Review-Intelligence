"""Public interface for priority calculation. No logic lives here.

Example:
    from app_review_intelligence.components.priority.interface import calculate_priorities

    result = calculate_priorities(
        {
            "subcategories": [{"subcategory_id": "PERF_001", "subcategory_name": "Application Crashes",
                               "description": "Unexpected shutdowns.", "major_category": "Performance and Stability"}],
            "assignments": [{"review_id": "R0001", "subcategory_id": "PERF_001"}],
        },
        [{"review_id": "R0001", "sentiment": "Negative", "confidence": 0.94}],
    )
    # [{"subcategory_id": "PERF_001", "subcategory_name": "Application Crashes", "review_count": 1,
    #   "negative_proportion": 1.0, "priority_score": ..., "priority_level": "...", "explanation": "...",
    #   "components": {...}}]
"""

from app_review_intelligence.components.priority.implementation import calculate_priority_results


def calculate_priorities(subcategories, sentiment_results, review_records=None, config=None):
    """
    Calculates a priority score and level for every discovered subcategory.

    Input:
        subcategories: subcategory result {"subcategories": [...], "assignments": [...]}.
        sentiment_results: [{"review_id", "sentiment", "confidence"}].
        review_records: optional model-facing records [{"review_id", "review_text", "review_date"}].
            Enables review-text severity matching and the trend component. Never pass gold labels.
        config: optional dict; {"priority": {...}} overrides config/settings.yaml,
            {"config_dir": path} picks another config folder.

    Returns:
        List of priority results (highest score first), one per subcategory, each with
        subcategory_id, subcategory_name, review_count, negative_proportion, priority_score,
        priority_level, explanation and a `components` object.

    Raises:
        TypeError / ValueError for malformed input, duplicate review IDs, unknown subcategory
        IDs, unknown sentiment labels or invalid priority settings.
    """
    return calculate_priority_results(subcategories, sentiment_results, review_records, config)
