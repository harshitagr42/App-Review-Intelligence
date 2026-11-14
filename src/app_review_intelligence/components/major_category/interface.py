"""Public interface for major-category classification. No logic lives here.

Example:
    from app_review_intelligence.components.major_category.interface import classify

    records = [
        {"review_id": "R0001", "review_text": "The app crashes whenever I try to log in.",
         "review_date": "2026-07-21"},
    ]
    classify(records)
    # [{"review_id": "R0001",
    #   "predicted_major_category": "Performance and Stability",
    #   "confidence": 0.71}]

Category names and the similarity threshold come from config/major_categories.yaml.
"""

from app_review_intelligence.components.major_category.implementation import (
    classify_reviews,
    release_model,
)


def classify(records):
    """
    Predicts one major category for every review.

    Input:
        records: list of model-facing review records
            {"review_id", "review_text", "review_date"}. Only review_text is used
            (review_id is just an identifier). Records with gold_* fields are rejected.

    Returns:
        List (same order as input, one item per review) of
        {"review_id", "predicted_major_category", "confidence"}.
        confidence is in [0, 1]. "Other" is returned when no category matches well.

    Raises:
        TypeError / ValueError: malformed records, repeated review_ids, gold labels
            present, or an invalid config/major_categories.yaml.
        FileNotFoundError: config/major_categories.yaml is missing.
    """
    return classify_reviews(records)


def unload_model():
    """Frees the MiniLM model (call before loading the sentiment model; never keep both)."""
    release_model()
