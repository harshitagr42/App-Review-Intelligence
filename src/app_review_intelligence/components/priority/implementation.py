"""Priority and severity calculation (Person 5).

Score for each discovered subcategory (0-100):

    score = w_volume * V + w_negative * N + w_severity * S + w_trend * T

V  volume: 100 * ln(1 + n) / ln(1 + n_max), where n_max is the largest group in the same major
   category (so one huge category cannot drown out the others).
N  negative sentiment: 100 * (negative reviews / reviews with a sentiment result).
S  severity: 25 / 50 / 75 / 100 from the phrase lists in config/settings.yaml. The highest
   matching level wins; nothing matched gives the lowest score.
T  trend: 50 means flat, 100 means complaints are rising fast, 0 means they are falling. Needs
   review dates; when it cannot be computed it is None and its weight is shared out over V, N, S.

Levels: score >= critical band -> Critical, >= high -> High, >= medium -> Medium, else Low.
UNASSIGNED reviews never count. Everything is deterministic and inputs are never modified.
"""

import copy
import logging
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

UNASSIGNED = "UNASSIGNED"
COMPONENT_NAMES = ("volume", "negative_sentiment", "severity", "trend")
SEVERITY_LEVELS = ("critical", "high", "medium", "low")
VALID_SENTIMENTS = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
UNSPECIFIED_MAJOR = "(unspecified)"

# Mirrors the `priority:` section of config/settings.yaml; used when the file or a key is missing.
DEFAULT_PRIORITY_SETTINGS = {
    "weights": {"volume": 0.35, "negative_sentiment": 0.35, "severity": 0.20, "trend": 0.10},
    "priority_bands": {"critical": 80, "high": 60, "medium": 35},
    "trend": {"recent_window_days": 30, "previous_window_days": 30, "minimum_reviews_per_window": 5},
    "severity": {
        "critical": {"score": 100, "phrases": [
            "account compromised", "data breach", "money taken without authorisation",
            "charged without authorisation", "permanent data loss"]},
        "high": {"score": 75, "phrases": [
            "cannot complete payment", "cannot log in", "refund never received", "account inaccessible"]},
        "medium": {"score": 50, "phrases": [
            "app keeps crashing", "subscription will not cancel", "feature not working"]},
        "low": {"score": 25, "phrases": [
            "layout is confusing", "button is misplaced", "text is too small"]},
    },
}


# ---------------------------------------------------------------- settings

def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_settings(settings):
    weights = settings["weights"]
    for name in COMPONENT_NAMES:
        if not _is_number(weights.get(name)) or weights[name] < 0:
            raise ValueError(f"priority.weights.{name} must be a number >= 0")
    if not any(weights[name] > 0 for name in ("volume", "negative_sentiment", "severity")):
        raise ValueError("priority.weights: volume, negative_sentiment and severity cannot all be 0")

    bands = settings["priority_bands"]
    for name in ("critical", "high", "medium"):
        if not _is_number(bands.get(name)):
            raise ValueError(f"priority.priority_bands.{name} must be a number")
    if not bands["critical"] > bands["high"] > bands["medium"]:
        raise ValueError("priority.priority_bands must satisfy critical > high > medium")

    trend = settings["trend"]
    for name in ("recent_window_days", "previous_window_days", "minimum_reviews_per_window"):
        if not isinstance(trend.get(name), int) or isinstance(trend.get(name), bool) or trend[name] < 1:
            raise ValueError(f"priority.trend.{name} must be an integer >= 1")

    for level in SEVERITY_LEVELS:
        entry = settings["severity"].get(level)
        if not isinstance(entry, dict) or not _is_number(entry.get("score")):
            raise ValueError(f"priority.severity.{level}.score must be a number")
        phrases = entry.get("phrases", [])
        if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
            raise ValueError(f"priority.severity.{level}.phrases must be a list of strings")


def load_priority_settings(config=None):
    """Defaults < config/settings.yaml `priority:` section < config["priority"] overrides."""
    config = config if isinstance(config, dict) else {}
    config_dir = config.get("config_dir")
    base = Path(config_dir) if config_dir else Path(__file__).resolve().parents[4] / "config"
    settings = copy.deepcopy(DEFAULT_PRIORITY_SETTINGS)

    settings_file = base / "settings.yaml"
    if settings_file.is_file():
        try:
            with open(settings_file, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as error:
            logger.warning("Could not read %s (%s); using built-in priority defaults.", settings_file, type(error).__name__)
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("priority"), dict):
            settings = _deep_merge(settings, loaded["priority"])

    if isinstance(config.get("priority"), dict):
        settings = _deep_merge(settings, config["priority"])

    _validate_settings(settings)
    return settings


# ---------------------------------------------------------------- input validation

def _validate_subcategory_result(result):
    if not isinstance(result, dict):
        raise TypeError("subcategories must be a dict with 'subcategories' and 'assignments'")
    subcategories = result.get("subcategories")
    assignments = result.get("assignments")
    if not isinstance(subcategories, list) or not isinstance(assignments, list):
        raise ValueError("subcategories must contain the lists 'subcategories' and 'assignments'")

    by_id = {}
    for item in subcategories:
        if not isinstance(item, dict):
            raise TypeError("every subcategory must be a dict")
        subcategory_id = item.get("subcategory_id")
        if not isinstance(subcategory_id, str) or not subcategory_id.strip():
            raise ValueError("every subcategory needs a non-empty subcategory_id")
        if subcategory_id == UNASSIGNED:
            continue
        if subcategory_id in by_id:
            raise ValueError(f"duplicate subcategory_id: {subcategory_id}")
        by_id[subcategory_id] = item

    reviews_by_subcategory = {subcategory_id: [] for subcategory_id in by_id}
    seen = set()
    for item in assignments:
        if not isinstance(item, dict):
            raise TypeError("every assignment must be a dict")
        if "review_id" not in item or "subcategory_id" not in item:
            raise ValueError("every assignment needs review_id and subcategory_id")
        review_id = str(item["review_id"])
        if review_id in seen:
            raise ValueError(f"review_id assigned more than once: {review_id}")
        seen.add(review_id)
        subcategory_id = item["subcategory_id"]
        if subcategory_id == UNASSIGNED:
            continue
        if subcategory_id not in by_id:
            raise ValueError(f"assignment refers to unknown subcategory_id: {subcategory_id}")
        reviews_by_subcategory[subcategory_id].append(review_id)
    return by_id, reviews_by_subcategory


def _sentiment_map(sentiment_results):
    if not isinstance(sentiment_results, list):
        raise TypeError("sentiment_results must be a list")
    mapping = {}
    for item in sentiment_results:
        if not isinstance(item, dict) or "review_id" not in item or "sentiment" not in item:
            raise ValueError("every sentiment result needs review_id and sentiment")
        review_id = str(item["review_id"])
        if review_id in mapping:
            raise ValueError(f"duplicate sentiment result for review_id: {review_id}")
        label = VALID_SENTIMENTS.get(str(item["sentiment"]).strip().lower())
        if label is None:
            raise ValueError(f"unknown sentiment label: {item['sentiment']!r}")
        mapping[review_id] = label
    return mapping


def _record_map(review_records):
    """Returns {review_id: {"text": str, "date": date|None}} from model-facing records only."""
    if review_records is None:
        return None
    if not isinstance(review_records, list):
        raise TypeError("review_records must be a list of dicts or None")
    mapping = {}
    for item in review_records:
        if not isinstance(item, dict):
            raise TypeError("every review record must be a dict")
        if "review_id" not in item:
            raise ValueError("every review record needs review_id")
        review_id = str(item["review_id"])
        if review_id in mapping:
            raise ValueError(f"duplicate review_id in review_records: {review_id}")
        text = item.get("review_text")
        mapping[review_id] = {
            "text": text if isinstance(text, str) else "",
            "date": _parse_date(item.get("review_date")),
        }
    return mapping


# ---------------------------------------------------------------- text and dates

_REPLACEMENTS = (
    (re.compile(r"\b(?:can't|cant|can not|cannot)\b"), "cannot"),
    (re.compile(r"\b(?:won't|wont|will not)\b"), "will not"),
    (re.compile(r"\blog[\s-]?in\b"), "log in"),
    (re.compile(r"authoriz"), "authoris"),
)
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalise(text):
    text = str(text).lower().replace("\u2019", "'").replace("\u2018", "'")
    for pattern, replacement in _REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return " " + _NON_WORD.sub(" ", text).strip() + " "


def _parse_date(value):
    try:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str) and value.strip():
            return date.fromisoformat(value.strip()[:10])
    except (ValueError, AttributeError):
        return None
    return None


# ---------------------------------------------------------------- components

def _volume_score(count, max_count):
    if count <= 0 or max_count <= 0:
        return 0.0
    return 100.0 * math.log1p(count) / math.log1p(max_count)


def _severity(subcategory, review_ids, records, severity_settings):
    """Returns (score, reason_codes, source). Highest matching level wins."""
    texts = [_normalise(f"{subcategory.get('subcategory_name', '')} {subcategory.get('description', '')}")]
    source = "subcategory_text"
    if records is not None:
        review_texts = [_normalise(records[r]["text"]) for r in review_ids if r in records and records[r]["text"]]
        if review_texts:
            texts.extend(review_texts)
            source = "subcategory_and_review_text"

    baseline = severity_settings["low"]["score"]
    best_score, codes, found = baseline, [], False
    for level in SEVERITY_LEVELS:  # most severe first
        entry = severity_settings[level]
        matched = [
            phrase for phrase in entry.get("phrases", [])
            if any(_normalise(phrase) in text for text in texts)
        ]
        codes.extend(f"{level}:{phrase}" for phrase in matched)
        if matched and not found:
            best_score, found = entry["score"], True
    if not codes:
        codes = ["no_phrase_match"]
    return best_score, codes, source


def _trend(review_ids, records, sentiments, reference_date, trend_settings):
    """Returns (score or None, status)."""
    if records is None:
        return None, "unavailable: review dates were not provided"
    if reference_date is None:
        return None, "unavailable: no valid review dates"

    recent_days = trend_settings["recent_window_days"]
    previous_days = trend_settings["previous_window_days"]
    minimum = trend_settings["minimum_reviews_per_window"]
    recent_start = reference_date - timedelta(days=recent_days - 1)
    previous_end = recent_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=previous_days - 1)

    total = {"recent": 0, "previous": 0}
    negative = {"recent": 0, "previous": 0}
    for review_id in review_ids:
        review_date = records.get(review_id, {}).get("date")
        if review_date is None:
            continue
        if recent_start <= review_date <= reference_date:
            window = "recent"
        elif previous_start <= review_date <= previous_end:
            window = "previous"
        else:
            continue
        total[window] += 1
        if sentiments.get(review_id) == "Negative":
            negative[window] += 1

    if total["recent"] < minimum or total["previous"] < minimum:
        return None, (
            f"insufficient_data: {total['recent']} reviews in the last {recent_days} days and "
            f"{total['previous']} in the {previous_days} days before (need at least {minimum} in each)"
        )

    recent_rate = negative["recent"] / recent_days
    previous_rate = negative["previous"] / previous_days
    peak = max(recent_rate, previous_rate)
    change = 0.0 if peak == 0 else (recent_rate - previous_rate) / peak
    score = 50.0 + 50.0 * change
    return score, (
        f"computed: {negative['recent']} negative reviews in the last {recent_days} days "
        f"vs {negative['previous']} in the {previous_days} days before"
    )


def _effective_weights(base_weights, trend_available):
    weights = {name: float(base_weights[name]) for name in COMPONENT_NAMES}
    if not trend_available:
        weights["trend"] = 0.0
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def _level(score, bands):
    if score >= bands["critical"]:
        return "Critical"
    if score >= bands["high"]:
        return "High"
    if score >= bands["medium"]:
        return "Medium"
    return "Low"


def _severity_name(score, severity_settings):
    for level in SEVERITY_LEVELS:
        if severity_settings[level]["score"] == score:
            return level
    return "custom"


def _explain(level, count, volume, negative_proportion, severity_name, codes, trend):
    if volume >= 75:
        volume_part = f"{count} reviews (high volume for its category)"
    elif volume >= 40:
        volume_part = f"{count} reviews (moderate volume for its category)"
    else:
        volume_part = f"{count} {'review' if count == 1 else 'reviews'} (low volume for its category)"

    share = round(100 * negative_proportion)
    if negative_proportion >= 0.6:
        sentiment_part = f"{share}% negative"
    elif negative_proportion <= 0.2:
        sentiment_part = f"only {share}% negative"
    else:
        sentiment_part = f"{share}% negative (mixed)"

    if codes == ["no_phrase_match"]:
        severity_part = "no severity phrases matched"
    else:
        severity_part = f"{severity_name} severity signal ({codes[0].split(':', 1)[1]})"

    parts = [volume_part, sentiment_part, severity_part]
    if trend is not None:
        parts.append("complaints rising" if trend >= 60 else "complaints falling" if trend <= 40 else "complaints stable")
    return f"{level} priority: " + "; ".join(parts) + "."


# ---------------------------------------------------------------- main entry

def calculate_priority_results(subcategory_result, sentiment_results, review_records=None, config=None):
    settings = load_priority_settings(config)
    subcategories, reviews_by_subcategory = _validate_subcategory_result(subcategory_result)
    sentiments = _sentiment_map(sentiment_results)
    records = _record_map(review_records)

    all_dates = [entry["date"] for entry in (records or {}).values() if entry["date"] is not None]
    reference_date = max(all_dates) if all_dates else None

    missing_sentiment = sum(
        1 for review_ids in reviews_by_subcategory.values() for r in review_ids if r not in sentiments
    )
    if missing_sentiment:
        logger.warning("%d assigned reviews have no sentiment result; they are left out of the negative share.", missing_sentiment)

    # Largest group per major category, for log-scaled volume.
    largest = {}
    for subcategory_id, subcategory in subcategories.items():
        major = subcategory.get("major_category") or UNSPECIFIED_MAJOR
        largest[major] = max(largest.get(major, 0), len(reviews_by_subcategory[subcategory_id]))

    results = []
    for subcategory_id, subcategory in subcategories.items():
        review_ids = reviews_by_subcategory[subcategory_id]
        count = len(review_ids)
        major = subcategory.get("major_category") or UNSPECIFIED_MAJOR

        with_sentiment = [r for r in review_ids if r in sentiments]
        negatives = sum(1 for r in with_sentiment if sentiments[r] == "Negative")
        negative_proportion = negatives / len(with_sentiment) if with_sentiment else 0.0

        volume = _volume_score(count, largest[major])
        negative_score = 100.0 * negative_proportion
        severity, codes, severity_source = _severity(subcategory, review_ids, records, settings["severity"])
        trend, trend_status = _trend(review_ids, records, sentiments, reference_date, settings["trend"])

        weights = _effective_weights(settings["weights"], trend is not None)
        values = {"volume": volume, "negative_sentiment": negative_score, "severity": severity, "trend": trend}
        raw_score = sum(weights[name] * (values[name] or 0.0) for name in COMPONENT_NAMES)
        score = round(min(max(raw_score, 0.0), 100.0), 2)
        level = _level(score, settings["priority_bands"])

        results.append({
            "subcategory_id": subcategory_id,
            "subcategory_name": subcategory.get("subcategory_name") or subcategory_id,
            "review_count": count,
            "negative_proportion": negative_proportion,
            "priority_score": score,
            "priority_level": level,
            "explanation": _explain(
                level, count, volume, negative_proportion,
                _severity_name(severity, settings["severity"]), codes, trend,
            ),
            "components": {
                "volume": volume,
                "negative_sentiment": negative_score,
                "severity": severity,
                "trend": trend,
                "effective_weights": weights,
                "severity_reason_codes": codes,
                "severity_source": severity_source,
                "trend_status": trend_status,
            },
        })

    results.sort(key=lambda r: (-r["priority_score"], -r["review_count"], r["subcategory_id"]))
    return results
