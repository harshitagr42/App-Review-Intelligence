"""
Priority scoring per subcategory (Person 5).

Rules: Docs/07_SUBCATEGORY_AND_PRIORITY_RULES.md, section 4.

    Priority = wV*V + wN*N + wS*S + wT*T   (clamped to 0-100)

- V (volume): 100 * log(1 + n) / log(1 + max_n), where max_n is the largest
  subcategory review count inside the same major category.
- N (negative sentiment): 100 * negative reviews / reviews in the subcategory.
- S (severity): fixed score for the subcategory's major category, from
  `priority.severity.category_scores` (100/75/50/25); `default_score` otherwise.
- T (trend): compares the subcategory's review counts in the most recent
  window with the preceding window. Both windows end relative to the latest
  review date in the whole run (not the wall clock), so results are
  reproducible. Growth g = (recent - previous) / previous is mapped to
  T = clamp(50 + 50 * g, 0, 100): 50 means flat, 100 means doubled or more,
  0 means the issue disappeared. T is unavailable (None) when dates are
  missing or either window has fewer than `minimum_reviews_per_window` reviews;
  its weight is then redistributed proportionally over V, N and S.

UNASSIGNED reviews are excluded from every group and from ranking.
Priority is a sorting heuristic, not a validated measure.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

UNASSIGNED = "UNASSIGNED"
COMPONENT_KEYS = ("volume", "negative_sentiment", "severity", "trend")
DEFAULT_SEVERITY = 25

# .../src/app_review_intelligence/components/priority/implementation.py -> repo root
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "config" / "settings.yaml"

DEFAULT_PRIORITY_CONFIG = {
    "weights": {"volume": 0.35, "negative_sentiment": 0.35, "severity": 0.20, "trend": 0.10},
    "priority_bands": {"critical": 80, "high": 60, "medium": 35},
    "trend": {"recent_window_days": 30, "previous_window_days": 30, "minimum_reviews_per_window": 5},
    "severity": {"default_score": DEFAULT_SEVERITY, "category_scores": {}},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_priority_config(config=None, settings_path=None):
    """
    Returns the `priority` section as a plain dict.

    `config` may be the priority section itself (or a full settings dict with a
    `priority` key). When omitted, config/settings.yaml is read; if that file is
    missing the documented defaults are used.
    """
    if config is None:
        path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
        if path.is_file():
            import yaml

            with path.open(encoding="utf-8") as handle:
                config = (yaml.safe_load(handle) or {}).get("priority") or {}
        else:
            config = {}
    elif "priority" in config and isinstance(config["priority"], dict):
        config = config["priority"]

    merged = {key: dict(value) for key, value in DEFAULT_PRIORITY_CONFIG.items()}
    for key, value in (config or {}).items():
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value
    _validate_config(merged)
    _validate_severity(merged["severity"])
    return merged


def _validate_config(config):
    weights = config["weights"]
    missing = [key for key in COMPONENT_KEYS if key not in weights]
    if missing:
        raise ValueError(f"priority.weights is missing {missing}")
    for key in COMPONENT_KEYS:
        if not isinstance(weights[key], (int, float)) or weights[key] < 0:
            raise ValueError(f"priority.weights.{key} must be a non-negative number")
    if not math.isclose(sum(weights[key] for key in COMPONENT_KEYS), 1.0, abs_tol=1e-6):
        raise ValueError("priority.weights must sum to 1.0")
    bands = config["priority_bands"]
    if not bands["critical"] > bands["high"] > bands["medium"] >= 0:
        raise ValueError("priority_bands must satisfy critical > high > medium >= 0")


def _validate_severity(severity):
    allowed = {25, 50, 75, 100}
    scores = dict(severity.get("category_scores") or {})
    scores["<default_score>"] = severity.get("default_score", DEFAULT_SEVERITY)
    for name, value in scores.items():
        if value not in allowed:
            raise ValueError(f"priority.severity score for {name!r} must be one of {sorted(allowed)}")


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _parse_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _volume(n, max_n):
    if n <= 0 or max_n <= 0:
        return 0.0
    return 100.0 * math.log(1 + n) / math.log(1 + max_n)


def _severity(major_category, severity_config):
    """Returns (score, reason_codes) from the major category's configured score."""
    scores = severity_config.get("category_scores") or {}
    if major_category in scores:
        return scores[major_category], [f"major_category: {major_category}"]
    return severity_config.get("default_score", DEFAULT_SEVERITY), []


def _trend(review_ids, date_by_id, anchor, trend_config):
    """Returns (score or None, status string, details dict)."""
    if anchor is None:
        return None, "unavailable: no valid review dates", {}
    recent_days = int(trend_config["recent_window_days"])
    previous_days = int(trend_config["previous_window_days"])
    minimum = int(trend_config["minimum_reviews_per_window"])

    recent_start = anchor - timedelta(days=recent_days - 1)
    previous_start = recent_start - timedelta(days=previous_days)
    recent = previous = 0
    for rid in review_ids:
        day = date_by_id.get(rid)
        if day is None:
            continue
        if recent_start <= day <= anchor:
            recent += 1
        elif previous_start <= day < recent_start:
            previous += 1
    details = {"recent_count": recent, "previous_count": previous}
    if recent < minimum or previous < minimum:
        return None, (
            f"unavailable: fewer than {minimum} dated reviews in a window "
            f"(recent {recent}, previous {previous})"
        ), details
    growth = (recent - previous) / previous
    score = min(max(50.0 + 50.0 * growth, 0.0), 100.0)
    return score, f"available: recent {recent} vs previous {previous} ({growth:+.0%})", details


def _effective_weights(weights, trend_available):
    if trend_available:
        return {key: float(weights[key]) for key in COMPONENT_KEYS}
    rest = sum(weights[key] for key in COMPONENT_KEYS if key != "trend")
    if rest <= 0:
        raise ValueError("priority weights for volume, negative_sentiment and severity sum to 0")
    effective = {key: weights[key] / rest for key in COMPONENT_KEYS if key != "trend"}
    effective["trend"] = 0.0
    return effective


def _level(score, bands):
    if score >= bands["critical"]:
        return "Critical"
    if score >= bands["high"]:
        return "High"
    if score >= bands["medium"]:
        return "Medium"
    return "Low"


def _explanation(name, major, n, negative, volume, severity, codes, trend_status, level):
    parts = [
        f"{n} review{'s' if n != 1 else ''} (volume {volume:.0f}/100 within {major or 'its category'})",
        f"{(negative / n if n else 0):.0%} negative",
    ]
    if codes:
        parts.append(f"severity {severity:g} for {major}")
    else:
        parts.append(f"severity {severity:g} (default; category not configured)")
    parts.append("trend " + trend_status)
    return f"{level} priority for '{name}': " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate_priorities(subcategory_result, sentiment_results, records=None, config=None):
    """
    See interface.calculate_priorities. Inputs are never mutated.
    Results are sorted by priority_score (desc), then subcategory_id.
    """
    if not isinstance(subcategory_result, dict):
        raise TypeError("subcategories must be the subcategory result dict "
                        "({'subcategories': [...], 'assignments': [...]})")
    priority_config = load_priority_config(config)

    subcategories = list(subcategory_result.get("subcategories") or [])
    assignments = list(subcategory_result.get("assignments") or [])
    sentiment_by_id = {
        item["review_id"]: item.get("sentiment")
        for item in sentiment_results or []
        if isinstance(item, dict) and "review_id" in item
    }

    date_by_id = {}
    for record in records or []:
        rid = record.get("review_id")
        if rid is None:
            continue
        parsed = _parse_date(record.get("review_date"))
        if parsed is not None:
            date_by_id[rid] = parsed
    anchor = max(date_by_id.values()) if date_by_id else None

    known_ids = {sub["subcategory_id"] for sub in subcategories}
    members = {sub_id: [] for sub_id in known_ids}
    for assignment in assignments:
        sub_id = assignment.get("subcategory_id")
        if sub_id == UNASSIGNED or sub_id not in members:
            continue
        members[sub_id].append(assignment["review_id"])

    max_by_major = {}
    for sub in subcategories:
        major = sub.get("major_category")
        max_by_major[major] = max(max_by_major.get(major, 0), len(members[sub["subcategory_id"]]))

    weights = priority_config["weights"]
    bands = priority_config["priority_bands"]
    results = []
    for sub in subcategories:
        sub_id = sub["subcategory_id"]
        review_ids = members[sub_id]
        n = len(review_ids)
        negative_ids = [rid for rid in review_ids if sentiment_by_id.get(rid) == "Negative"]
        negative_proportion = len(negative_ids) / n if n else 0.0

        volume = _volume(n, max_by_major[sub.get("major_category")])
        negative = 100.0 * negative_proportion
        severity, codes = _severity(sub.get("major_category"), priority_config["severity"])
        trend, trend_status, trend_details = _trend(review_ids, date_by_id, anchor, priority_config["trend"])

        effective = _effective_weights(weights, trend is not None)
        values = {"volume": volume, "negative_sentiment": negative, "severity": severity, "trend": trend}
        raw = sum(effective[key] * (values[key] or 0.0) for key in COMPONENT_KEYS)
        score = round(min(max(raw, 0.0), 100.0), 2)
        level = _level(score, bands)

        results.append({
            "subcategory_id": sub_id,
            "subcategory_name": sub.get("subcategory_name"),
            "review_count": n,
            "negative_proportion": negative_proportion,
            "priority_score": score,
            "priority_level": level,
            "explanation": _explanation(
                sub.get("subcategory_name"), sub.get("major_category"), n, len(negative_ids),
                volume, severity, codes, trend_status, level,
            ),
            "components": {
                "volume": volume,
                "negative_sentiment": negative,
                "severity": severity,
                "trend": trend,
                "effective_weights": effective,
                "severity_reason_codes": codes,
                "trend_status": trend_status,
                "trend_counts": trend_details,
            },
        })

    results.sort(key=lambda item: (-item["priority_score"], item["subcategory_id"]))
    return results
