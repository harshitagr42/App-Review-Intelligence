"""Sentiment prediction with a pretrained CardiffNLP RoBERTa model (Person 3).

Works on model-facing records only: {"review_id", "review_text", "review_date"}.
Only ``review_text`` is fed to the model. ``review_id`` is an identifier and
``review_date`` is ignored. Gold labels never reach this module.

The model (cardiffnlp/twitter-roberta-base-sentiment-latest) is loaded once,
lazily, and reused for every call. No training or fine-tuning happens here.
"""

import gc
import logging
import threading

logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 8
MAX_LENGTH = 512
SENTIMENT_LABELS = ("Positive", "Negative", "Neutral")

_LABEL_NORMALISATION = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}

_lock = threading.Lock()
_bundle = None  # (tokenizer, model, id2label) once loaded


class SentimentModelError(RuntimeError):
    """The sentiment model could not be loaded."""


class SentimentPredictionError(RuntimeError):
    """The model failed while predicting a batch."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_records(records):
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

        text = record.get("review_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"record {review_id!r} has empty or non-string review_text"
            )


# ---------------------------------------------------------------------------
# Model loading (lazy, thread-safe, once per process)
# ---------------------------------------------------------------------------

def _build_label_map(raw_id2label):
    """Map model class indices to Positive/Negative/Neutral using the model's own config."""
    label_map = {}
    for index, name in raw_id2label.items():
        key = str(name).strip().lower()
        if key not in _LABEL_NORMALISATION:
            raise SentimentModelError(
                f"Unexpected label {name!r} in model config; expected "
                "negative/neutral/positive."
            )
        label_map[int(index)] = _LABEL_NORMALISATION[key]

    if set(label_map.values()) != set(SENTIMENT_LABELS):
        raise SentimentModelError(
            f"Model labels {sorted(label_map.values())} do not match {sorted(SENTIMENT_LABELS)}."
        )
    return label_map


def _load_model():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    except Exception as error:
        raise SentimentModelError(
            f"Could not load sentiment model {MODEL_NAME!r}: {error}"
        ) from error

    model.eval()
    id2label = _build_label_map(model.config.id2label)
    logger.info("Loaded sentiment model %s", MODEL_NAME)
    return tokenizer, model, id2label


def _get_model():
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                _bundle = _load_model()
    return _bundle


def unload_model():
    """Free the model (the project never keeps both local models in memory)."""
    global _bundle
    with _lock:
        _bundle = None
    gc.collect()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _scores_to_predictions(logits, id2label):
    """Softmax over class logits -> [(label, confidence)] using the top probability."""
    import torch

    probabilities = torch.softmax(logits, dim=-1)
    confidences, indices = probabilities.max(dim=-1)
    return [
        (id2label[int(index)], min(max(float(confidence), 0.0), 1.0))
        for confidence, index in zip(confidences, indices)
    ]


def _classify_batch(texts):
    """Run one batch through the model. Returns [(label, confidence)] in input order."""
    tokenizer, model, id2label = _get_model()
    try:
        import torch

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
        return _scores_to_predictions(logits, id2label)
    except Exception as error:
        raise SentimentPredictionError(
            f"Sentiment prediction failed for a batch of {len(texts)}: {error}"
        ) from error


def predict_sentiments(records):
    """Predict exactly one sentiment per record.

    Returns:
        [{"review_id", "sentiment", "confidence"}] in the same order as ``records``.

    Raises:
        TypeError / ValueError: malformed records, missing/duplicate review_id,
            empty text.
        SentimentModelError: model could not be loaded.
        SentimentPredictionError: inference failed.
    """
    _validate_records(records)
    if not records:
        return []

    results = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        predictions = _classify_batch([record["review_text"] for record in batch])
        if len(predictions) != len(batch):
            raise SentimentPredictionError(
                f"Model returned {len(predictions)} predictions for {len(batch)} reviews."
            )
        for record, (label, confidence) in zip(batch, predictions):
            results.append(
                {
                    "review_id": record["review_id"],
                    "sentiment": label,
                    "confidence": confidence,
                }
            )
    return results