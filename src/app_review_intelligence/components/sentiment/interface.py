"""Public interface for sentiment analysis. No logic lives here.

Example:
    from app_review_intelligence.components.sentiment.interface import analyze

    analyze([{"review_id": "R0001", "review_text": "App keeps crashing.", "review_date": "2026-07-21"}])
    # [{"review_id": "R0001", "sentiment": "Negative", "confidence": 0.94}]
"""

from app_review_intelligence.components.sentiment.implementation import (
    MODEL_NAME,
    predict_sentiments,
    unload_model as _unload_model,
)


def analyze(records):
    """
    Predicts one sentiment for every review.

    Input:
        records: list of {"review_id", "review_text", "review_date"} dicts
            (review_date is optional and unused).

    Returns:
        [{"review_id", "sentiment": "Positive"|"Negative"|"Neutral",
          "confidence": float in [0, 1]}], same order as the input.

    Raises:
        TypeError / ValueError for malformed records, missing or duplicate
        review_id, or empty review_text.
    """
    return predict_sentiments(records)


def unload_model():
    """Frees the sentiment model (call before MiniLM is loaded again; never keep both)."""
    _unload_model()


def model_id():
    """Returns the sentiment model identifier used by analyze()."""
    return MODEL_NAME
