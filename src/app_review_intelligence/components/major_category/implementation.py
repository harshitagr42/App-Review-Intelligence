"""Major-category classification logic (Person 2).

Method (no training or fine-tuning):
    1. Load the category definitions from ``config/major_categories.yaml``.
    2. Embed every definition text (description + examples) and every review with
       ``sentence-transformers/all-MiniLM-L6-v2`` (L2-normalised vectors).
    3. A category's score for a review is the MAX cosine similarity over its texts.
    4. The best-scoring category wins; if its score is below ``min_similarity`` the
       review gets the fallback category ("Other").

Confidence:
    * matched category: the winning cosine similarity, clipped to [0, 1].
    * "Other": ``1 - best_score / min_similarity``, clipped to [0, 1]
      (the further below the threshold, the surer the review fits nothing).

Optional accuracy options (all off by default; set them in config/major_categories.yaml):
    * ``model``           - any sentence-transformers model name.
    * ``scoring``         - "max" (default), "top2" or "centroid": how a category's definition
                            texts are combined. "top2"/"centroid" are not biased toward
                            categories that happen to have more examples.
    * ``sentence_level``  - score each sentence of a review and keep the best one, so filler
                            sentences do not dilute the topic.
    * ``smoothing``       - {neighbors, weight}: blend each review's category scores with those
                            of its most similar reviews in the SAME call. Label-free, but a
                            review's result then depends on the other reviews passed in, and it
                            is skipped when the batch has no more than ``neighbors`` usable reviews.

    * ``other_examples``  - optional list of example texts for "Other". When given, "Other" competes
                            as a real category (its score is compared with the others), and
                            ``min_similarity`` still acts as a floor. ``other_bias`` (default 0)
                            is added to the "Other" score to make it win more or less often.

Only ``review_id`` (as an identifier) and ``review_text`` are read. Records carrying
``gold_*`` label fields are rejected so labels can never influence a prediction.
"""

import gc
import re
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
# .../src/app_review_intelligence/components/major_category/implementation.py -> repo root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "major_categories.yaml"

SCORING_METHODS = ("max", "top2", "centroid")
NEIGHBOR_SHARPNESS = 3  # neighbour weight = similarity ** this, so near-duplicates count most

_MODEL_CACHE = {}


def load_category_config(config_path=None):
    """Read and validate the YAML definitions. Returns a plain dict."""
    import yaml

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Major-category config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    categories = raw.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{path}: 'categories' must be a non-empty list")

    other = raw.get("other_category", "Other")
    threshold = raw.get("min_similarity", 0.30)
    if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError(f"{path}: 'min_similarity' must be a number in (0, 1]")

    cleaned, seen, other_entry = [], set(), None
    for entry in categories:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: every category needs a non-empty 'name'")
        if name == other and other_entry is None:
            other_entry = entry  # "Other" is listed in the taxonomy; its examples compete as "Other"
            continue
        if name in seen or name == other:
            raise ValueError(f"{path}: duplicate or reserved category name {name!r}")
        seen.add(name)
        texts = [str(entry.get("description") or "").strip()]
        texts += [str(e).strip() for e in entry.get("examples") or []]
        texts = [t for t in texts if t]
        if not texts:
            raise ValueError(f"{path}: category {name!r} has no description or examples")
        cleaned.append({"name": name, "texts": texts})

    model = raw.get("model", MODEL_NAME)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{path}: 'model' must be a non-empty string")

    scoring = raw.get("scoring", "max")
    if scoring not in SCORING_METHODS:
        raise ValueError(f"{path}: 'scoring' must be one of {SCORING_METHODS}")

    sentence_level = raw.get("sentence_level", False)
    if not isinstance(sentence_level, bool):
        raise ValueError(f"{path}: 'sentence_level' must be true or false")

    smoothing = raw.get("smoothing")
    if smoothing is not None:
        if not isinstance(smoothing, dict):
            raise ValueError(f"{path}: 'smoothing' must be a mapping with neighbors and weight")
        neighbors = smoothing.get("neighbors", 10)
        weight = smoothing.get("weight", 0.5)
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError(f"{path}: smoothing 'neighbors' must be an integer >= 1")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight < 1:
            raise ValueError(f"{path}: smoothing 'weight' must be a number in [0, 1)")
        smoothing = {"neighbors": neighbors, "weight": float(weight)}

    other_examples = raw.get("other_examples")
    if other_entry is not None:
        if other_examples is not None:
            raise ValueError(f"{path}: give the 'Other' examples under its category entry, not 'other_examples'")
        other_examples = other_entry.get("examples") or None
    if other_examples is not None:
        if not isinstance(other_examples, list) or not other_examples:
            raise ValueError(f"{path}: 'other_examples' must be a non-empty list of texts")
        other_examples = [str(e).strip() for e in other_examples if str(e).strip()]
        if not other_examples:
            raise ValueError(f"{path}: 'other_examples' must contain at least one non-empty text")

    other_bias = raw.get("other_bias", 0.0)
    if isinstance(other_bias, bool) or not isinstance(other_bias, (int, float)):
        raise ValueError(f"{path}: 'other_bias' must be a number")

    return {
        "categories": cleaned,
        "other_category": other,
        "min_similarity": float(threshold),
        "model": model.strip(),
        "scoring": scoring,
        "sentence_level": sentence_level,
        "smoothing": smoothing,
        "other_examples": other_examples,
        "other_bias": float(other_bias),
    }


def _get_default_embedder(model_name=None):
    """Lazily load a sentence-transformers model (cached by name); return embed(texts) -> ndarray."""
    name = model_name or MODEL_NAME
    model = _MODEL_CACHE.get(name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name)
        _MODEL_CACHE[name] = model

    def embed(texts):
        return model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return embed


def release_model():
    """Unload MiniLM so another local model (sentiment) can be loaded alone."""
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _validate_records(records):
    if not isinstance(records, list):
        raise TypeError("records must be a list of dicts")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"record {index} must be a dict")
        for key in ("review_id", "review_text"):
            if key not in record:
                raise ValueError(f"record {index} is missing '{key}'")
        leaked = [key for key in record if str(key).startswith("gold_")]
        if leaked:
            raise ValueError(f"record {index} contains ground-truth fields {leaked}")
        if record["review_text"] is not None and not isinstance(record["review_text"], str):
            raise TypeError(f"record {index}: 'review_text' must be a string")
        if record["review_id"] in seen:
            raise ValueError(f"duplicate review_id {record['review_id']!r}")
        seen.add(record["review_id"])


def _split_sentences(text):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if len(p.strip()) > 3]
    return parts or [text]


def _category_scores(vecs, def_vecs, owners, n_cat, scoring):
    """(n, n_cat) score of each vector against each category's definition texts."""
    import numpy as np

    if scoring == "centroid":
        centroids = np.stack([def_vecs[owners == c].mean(axis=0) for c in range(n_cat)])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        return vecs @ (centroids / np.where(norms == 0, 1.0, norms)).T
    similarity = vecs @ def_vecs.T
    columns = []
    for c in range(n_cat):
        top = np.sort(similarity[:, owners == c], axis=1)[:, ::-1]
        k = 1 if scoring == "max" else min(2, top.shape[1])
        columns.append(top[:, :k].mean(axis=1))
    return np.stack(columns, axis=1)


def _neighbors(vecs, k):
    """For each vector: indices and normalised weights of its k most similar OTHER vectors."""
    import numpy as np

    n = len(vecs)
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=int)
    weights = np.empty((n, k))
    for start in range(0, n, 1024):
        block = vecs[start:start + 1024] @ vecs.T
        rows = np.arange(block.shape[0])
        block[rows, start + rows] = -np.inf  # a review is not its own neighbour
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        idx[start:start + len(rows)] = part
        weights[start:start + len(rows)] = np.clip(np.take_along_axis(block, part, axis=1), 0, None) ** NEIGHBOR_SHARPNESS
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.where(totals > 0, weights / np.where(totals > 0, totals, 1.0), 1.0 / k)
    return idx, weights


def _smooth(scores, idx, weights, weight):
    """Blend each review's scores with the weighted average of its neighbours' scores."""
    neighbour_scores = (scores[idx] * weights[:, :, None]).sum(axis=1)
    return (1 - weight) * scores + weight * neighbour_scores


def classify_reviews(records, embed_fn=None, config=None):
    """
    Predict exactly one major category per review, preserving input order.

    Args:
        records: list of {"review_id", "review_text", ...} dicts.
        embed_fn: optional ``texts -> L2-normalised ndarray`` (used by tests to avoid
            loading the real model). Defaults to MiniLM.
        config: optional dict from ``load_category_config``.

    Returns:
        [{"review_id", "predicted_major_category", "confidence"}]
    """
    import numpy as np

    _validate_records(records)
    if not records:
        return []

    config = config or load_category_config()
    other, threshold = config["other_category"], config["min_similarity"]

    # Reviews with no usable text cannot match anything -> fallback, zero confidence.
    texts = [(r["review_text"] or "").strip() for r in records]
    usable = [i for i, text in enumerate(texts) if text]

    results = [
        {"review_id": r["review_id"], "predicted_major_category": other, "confidence": 0.0}
        for r in records
    ]
    if not usable:
        return results

    scoring = config.get("scoring", "max")
    sentence_level = config.get("sentence_level", False)
    smoothing = config.get("smoothing")
    embed = embed_fn or _get_default_embedder(config.get("model", MODEL_NAME))

    # Embed all definition texts once, remembering which category each belongs to.
    definition_texts, owners = [], []
    for category_index, category in enumerate(config["categories"]):
        definition_texts.extend(category["texts"])
        owners.extend([category_index] * len(category["texts"]))
    other_examples = config.get("other_examples")
    n_named = len(config["categories"])
    if other_examples:  # "Other" competes as one extra category, always the last column
        definition_texts.extend(other_examples)
        owners.extend([n_named] * len(other_examples))
    owners = np.array(owners)
    n_cat = n_named + (1 if other_examples else 0)
    definition_vectors = np.asarray(embed(definition_texts), dtype=float)

    usable_texts = [texts[i] for i in usable]
    use_smoothing = bool(smoothing) and len(usable) > smoothing["neighbors"]
    review_vectors = None
    if not sentence_level or use_smoothing:
        review_vectors = np.asarray(embed(usable_texts), dtype=float)

    if sentence_level:
        sentences, starts = [], []
        for text in usable_texts:
            starts.append(len(sentences))
            sentences.extend(_split_sentences(text))
        sentence_vectors = np.asarray(embed(sentences), dtype=float)
        sentence_scores = _category_scores(sentence_vectors, definition_vectors, owners, n_cat, scoring)
        scores = np.maximum.reduceat(sentence_scores, np.array(starts), axis=0)
    else:
        scores = _category_scores(review_vectors, definition_vectors, owners, n_cat, scoring)

    if use_smoothing:
        idx, weights = _neighbors(review_vectors, smoothing["neighbors"])
        scores = _smooth(scores, idx, weights, smoothing["weight"])

    if other_examples:
        scores = scores.copy()
        scores[:, n_named] += config.get("other_bias", 0.0)
    best = scores.argmax(axis=1)

    for row, record_index in enumerate(usable):
        best_score = float(scores[row, best[row]])
        if other_examples and best[row] == n_named:
            label = other
            confidence = min(max(best_score, 0.0), 1.0)
        elif best_score >= threshold:
            label = config["categories"][best[row]]["name"]
            confidence = min(max(best_score, 0.0), 1.0)
        else:
            label = other
            confidence = min(max(1.0 - best_score / threshold, 0.0), 1.0)
        results[record_index] = {
            "review_id": records[record_index]["review_id"],
            "predicted_major_category": label,
            "confidence": round(confidence, 4),
        }
    return results
