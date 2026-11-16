"""Reporting (Person 6): report.md, classified_reviews.csv, clusters.json, run_metadata.json.

Spec: Docs/09_CLI_AND_REPORT_SPECIFICATION.md section 5, Docs/05_DATA_SCHEMA.md 4-6.

Rules:
- Every file is created exclusively ("x" mode): nothing is ever overwritten.
- Gold-label fields are never exported.
- The Gemini API key is scrubbed from every artifact.
- Priority is a transparent heuristic, never described as accurate or validated.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
UNASSIGNED = "UNASSIGNED"
SNIPPET_LENGTH = 160
REPRESENTATIVE_COUNT = 3
SENTIMENTS = ("Negative", "Neutral", "Positive")

GOLD_FIELDS = {"gold_major_category", "gold_minor_category", "gold_sentiment"}

CSV_FIELDS = [
    "review_id",
    "source_file",
    "source_review_id",
    "source_row_number",
    "review_text",
    "review_date",
    "predicted_major_category",
    "major_category_confidence",
    "subcategory_id",
    "subcategory_name",
    "sentiment",
    "sentiment_confidence",
    "priority_score",
    "priority_level",
    "subcategory_error",
]

MODEL_ATTRIBUTION = [
    ("Embeddings and semantic similarity", "sentence-transformers/all-MiniLM-L6-v2", "Apache-2.0"),
    ("Sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest", "CC-BY-4.0, by Cardiff NLP"),
    ("Subcategory discovery", "gemini-3.6-flash", "Google Gemini API"),
]


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def _scrub_secrets(value: Any) -> Any:
    """Remove API keys from every string inside value."""
    if isinstance(value, dict):
        return {key: _scrub_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_secrets(item) for item in value]
    if not isinstance(value, str):
        return value

    result = value
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        result = result.replace(api_key, "[REDACTED_API_KEY]")
    result = re.sub(r"(?i)(GEMINI_API_KEY\s*[:=]\s*)['\"]?[^'\"\s,;]+", r"\1[REDACTED]", result)
    result = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)['\"]?[^'\"\s,;]+", r"\1[REDACTED]", result)
    return result


def _json_safe(value: Any) -> Any:
    """Convert to strict-JSON values: NaN/inf -> None, Paths -> str, tuples -> lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    serialized = json.dumps(
        _json_safe(_scrub_secrets(data)), indent=2, ensure_ascii=False, allow_nan=False,
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _write_csv(path: Path, records: list[dict]) -> None:
    if not isinstance(records, list):
        raise TypeError("classified_reviews must be a list.")
    seen = set()
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"classified review {index} must be a dictionary.")
        review_id = str(record.get("review_id") or "").strip()
        if not review_id:
            raise ValueError(f"classified review {index} has an empty review_id.")
        if review_id in seen:
            raise ValueError(f"duplicate review_id in classified reviews: {review_id}")
        seen.add(review_id)
        rows.append({
            key: _scrub_secrets(value)
            for key, value in record.items()
            if key not in GOLD_FIELDS
        })

    extra = [key for row in rows for key in row if key not in CSV_FIELDS]
    fieldnames = CSV_FIELDS + list(dict.fromkeys(extra))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _unique_report_path(output_dir: Path) -> Path:
    """Reserve report.md, or report_2.md, report_3.md ... if it already exists."""
    counter = 1
    while True:
        candidate = output_dir / ("report.md" if counter == 1 else f"report_{counter}.md")
        try:
            with candidate.open("x", encoding="utf-8"):
                pass
            return candidate
        except FileExistsError:
            counter += 1


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "n/a"
    return str(value)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "0.0%"


def _snippet(text: Any) -> str:
    clean = " ".join(str(text or "").split()).replace("|", "/")
    return clean if len(clean) <= SNIPPET_LENGTH else clean[: SNIPPET_LENGTH - 1].rstrip() + "…"


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


# ---------------------------------------------------------------------------
# clusters.json
# ---------------------------------------------------------------------------

def build_clusters(
    subcategory_result: dict[str, Any],
    classified_reviews: list[dict[str, Any]],
    priorities: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    llm_model_id: str | None = None,
) -> dict[str, Any]:
    """Build the clusters.json structure (Docs/05 section 5) from the subcategory result."""
    rows_by_id = {str(row.get("review_id")): row for row in classified_reviews}
    priority_by_id = {item.get("subcategory_id"): item for item in priorities or []}

    members: dict[str, list[str]] = defaultdict(list)
    unassigned: list[str] = []
    for assignment in subcategory_result.get("assignments", []):
        review_id = str(assignment.get("review_id"))
        subcategory_id = assignment.get("subcategory_id") or UNASSIGNED
        if subcategory_id == UNASSIGNED:
            unassigned.append(review_id)
        else:
            members[subcategory_id].append(review_id)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subcategory in subcategory_result.get("subcategories", []):
        subcategory_id = subcategory.get("subcategory_id")
        member_ids = members.get(subcategory_id, [])
        member_rows = [rows_by_id[i] for i in member_ids if i in rows_by_id]
        counts = Counter(row.get("sentiment") for row in member_rows)
        sentiment_counts = {label.lower(): counts.get(label, 0) for label in SENTIMENTS}
        # Representatives: the members the major-category model was most confident about.
        ranked = sorted(
            member_rows,
            key=lambda row: (-_number(row.get("major_category_confidence")), str(row.get("review_id"))),
        )
        grouped[subcategory.get("major_category") or "Other"].append({
            "subcategory_id": subcategory_id,
            "subcategory_name": subcategory.get("subcategory_name"),
            "description": subcategory.get("description"),
            "major_category": subcategory.get("major_category"),
            "member_review_ids": member_ids,
            "review_count": len(member_ids),
            "representative_review_ids": [
                str(row.get("review_id")) for row in ranked[:REPRESENTATIVE_COUNT]
            ],
            "sentiment_counts": sentiment_counts,
            "negative_proportion": (
                round(sentiment_counts["negative"] / len(member_rows), 4) if member_rows else 0.0
            ),
            "priority": priority_by_id.get(subcategory_id),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "llm_model_id": llm_model_id,
        "major_categories": [
            {"major_category": name, "subcategories": grouped[name]}
            for name in sorted(grouped)
        ],
        "unassigned_review_ids": unassigned,
    }


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

def _build_report(
    results: dict[str, Any],
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    clusters: dict[str, Any] | None,
) -> str:
    total = len(records)
    rows_by_id = {str(row.get("review_id")): row for row in records}
    priorities = sorted(
        results.get("priorities") or [],
        key=lambda item: -_number(item.get("priority_score")),
    )
    subcategories = [
        sub for group in (clusters or {}).get("major_categories", []) for sub in group["subcategories"]
    ]
    unassigned_ids = (clusters or {}).get("unassigned_review_ids") or results.get("unassigned_review_ids") or [
        str(row.get("review_id")) for row in records if row.get("subcategory_id") == UNASSIGNED
    ]
    major_counts = Counter(row.get("predicted_major_category") for row in records)
    sentiment_counts = Counter(row.get("sentiment") for row in records)
    warnings = list(results.get("warnings") or metadata.get("warnings") or [])
    lines: list[str] = []
    add = lines.append

    # 1. Title and run metadata
    add("# App Review Intelligence — Run Report")
    add("")
    add(f"- Run ID: {_fmt(metadata.get('run_id'))}")
    add(f"- Started (UTC): {_fmt(metadata.get('started_at_utc'))}")
    add(f"- Duration so far (s): {_fmt(metadata.get('duration_seconds'))}")
    add(f"- Random seed: {_fmt(metadata.get('random_seed'))}")
    add(f"- Settings SHA-256: {_fmt(metadata.get('settings_sha256'))}")
    add("")

    # 2. Executive summary
    add("## 1. Executive Summary")
    add("")
    add(f"- Reviews analysed: {total}")
    add(f"- Subcategories discovered: {len(subcategories)}")
    add(f"- Unassigned reviews: {len(unassigned_ids)} ({_pct(len(unassigned_ids), total)})")
    if major_counts:
        top_category, top_count = major_counts.most_common(1)[0]
        add(f"- Largest major category: {top_category} ({top_count} reviews, {_pct(top_count, total)})")
    add(
        "- Sentiment: " + ", ".join(
            f"{label} {sentiment_counts.get(label, 0)} ({_pct(sentiment_counts.get(label, 0), total)})"
            for label in SENTIMENTS
        )
    )
    if priorities:
        add("- Highest-priority issues (heuristic): " + "; ".join(
            f"{item.get('subcategory_name')} ({_fmt(item.get('priority_level'))}, "
            f"{_fmt(item.get('priority_score'), 1)})"
            for item in priorities[:3]
        ))
    else:
        add("- Priority scores: not available in this run (the priority component did not produce results).")
    if warnings:
        add(f"- Warnings: {len(warnings)} (see section 2 and logs.txt)")
    add("")

    # 3. Input summary
    add("## 2. Input Summary")
    add("")
    for item in metadata.get("input_files") or []:
        if isinstance(item, dict):
            add(f"- File: {item.get('file_name')} (SHA-256 `{_fmt(item.get('sha256'))}`)")
        else:
            add(f"- File: {item}")
    mapping = metadata.get("column_mapping") or {}
    if mapping:
        add(f"- Column mapping: ID=`{mapping.get('id')}`, text=`{mapping.get('text')}`, date=`{mapping.get('date')}`")
    add(f"- Rows read: {_fmt(metadata.get('rows_read'))}")
    add(f"- Rows accepted: {_fmt(metadata.get('rows_accepted'))}")
    add(f"- Removed, empty review: {_fmt(metadata.get('rows_skipped_empty'))}")
    add(f"- Removed, empty or invalid date: {_fmt(metadata.get('rows_skipped_invalid_date'))}")
    add(f"- Removed, duplicate text and date: {_fmt(metadata.get('duplicate_review_count'))}")
    for message in warnings[:20]:
        add(f"- Warning: {_scrub_secrets(str(message))}")
    if len(warnings) > 20:
        add(f"- … {len(warnings) - 20} more warnings in run_metadata.json")
    add("")

    # 4. Major-category distribution
    add("## 3. Major-Category Distribution")
    add("")
    if major_counts:
        subcategory_counts = Counter(sub.get("major_category") for sub in subcategories)
        add("| Major category | Reviews | Share | Subcategories |")
        add("|---|---:|---:|---:|")
        for name, count in sorted(major_counts.items(), key=lambda pair: (-pair[1], str(pair[0]))):
            add(f"| {name} | {count} | {_pct(count, total)} | {subcategory_counts.get(name, 0)} |")
    else:
        add("No classified reviews were supplied.")
    add("")

    # 5. Priority ranking
    add("## 4. Priority Ranking")
    add("")
    add(
        "Priority is a transparent heuristic (volume, negative sentiment, severity and trend). "
        "It has no ground truth and is not a measure of predictive performance."
    )
    add("")
    if priorities:
        names = {sub.get("subcategory_id"): sub.get("major_category") for sub in subcategories}
        add("| Rank | Subcategory | Major category | Reviews | Negative share | Score | Level |")
        add("|---:|---|---|---:|---:|---:|---|")
        for rank, item in enumerate(priorities, start=1):
            add(
                f"| {rank} | {item.get('subcategory_name')} (`{item.get('subcategory_id')}`) "
                f"| {_fmt(names.get(item.get('subcategory_id')))} | {_fmt(item.get('review_count'))} "
                f"| {_fmt(item.get('negative_proportion'))} | {_fmt(item.get('priority_score'), 1)} "
                f"| {_fmt(item.get('priority_level'))} |"
            )
    else:
        add("Priority scores are not available for this run.")
    add("")

    # 6. Detailed subcategory analysis grouped under major categories
    add("## 5. Detailed Subcategory Analysis")
    add("")
    if not subcategories:
        add("No subcategories were discovered.")
        add("")
    for group in (clusters or {}).get("major_categories", []):
        add(f"### {group['major_category']}")
        add("")
        for sub in sorted(group["subcategories"], key=lambda s: (-s["review_count"], str(s["subcategory_id"]))):
            count = sub["review_count"]
            counts = sub.get("sentiment_counts") or {}
            add(f"#### {sub.get('subcategory_name')} (`{sub.get('subcategory_id')}`)")
            add("")
            add(f"- Description: {_fmt(sub.get('description'))}")
            add(f"- Parent major category: {sub.get('major_category')}")
            add(f"- Reviews: {count}")
            add(
                "- Sentiment: " + ", ".join(
                    f"{label} {counts.get(label.lower(), 0)} ({_pct(counts.get(label.lower(), 0), count)})"
                    for label in SENTIMENTS
                )
            )
            priority = sub.get("priority")
            if priority:
                components = priority.get("components") or {}
                weights = components.get("effective_weights") or {}
                add(
                    f"- Priority: {_fmt(priority.get('priority_score'), 1)} "
                    f"({_fmt(priority.get('priority_level'))}) — {_fmt(priority.get('explanation'))}"
                )
                add(
                    f"- Components: V={_fmt(components.get('volume'), 1)}, "
                    f"N={_fmt(components.get('negative_sentiment'), 1)}, "
                    f"S={_fmt(components.get('severity'), 1)}, T={_fmt(components.get('trend'), 1)}"
                )
                add("- Effective weights: " + (", ".join(
                    f"{key}={_fmt(value, 4)}" for key, value in weights.items()
                ) or "n/a"))
                add("- Severity reason codes: " + (
                    ", ".join(str(code) for code in components.get("severity_reason_codes") or []) or "none"
                ))
                add(f"- Trend: {_fmt(components.get('trend_status'))}")
            else:
                add("- Priority: not available in this run")
            add("- Representative reviews:")
            for review_id in sub.get("representative_review_ids") or []:
                add(f"  - `{review_id}`: {_snippet(rows_by_id.get(review_id, {}).get('review_text'))}")
            add("")

    # 7. Sentiment distribution
    add("## 6. Sentiment Distribution")
    add("")
    add("| Sentiment | Reviews | Share |")
    add("|---|---:|---:|")
    for label in SENTIMENTS:
        add(f"| {label} | {sentiment_counts.get(label, 0)} | {_pct(sentiment_counts.get(label, 0), total)} |")
    add("")

    # 8. Unassigned reviews summary
    add("## 7. Unassigned Reviews Summary")
    add("")
    add(f"- Unassigned reviews: {len(unassigned_ids)} ({_pct(len(unassigned_ids), total)})")
    by_category = Counter(
        rows_by_id.get(str(review_id), {}).get("predicted_major_category") for review_id in unassigned_ids
    )
    for name, count in sorted(by_category.items(), key=lambda pair: (-pair[1], str(pair[0]))):
        add(f"- {name}: {count}")
    if unassigned_ids:
        shown = ", ".join(f"`{review_id}`" for review_id in unassigned_ids[:20])
        more = f" (+{len(unassigned_ids) - 20} more in clusters.json)" if len(unassigned_ids) > 20 else ""
        add(f"- Review IDs: {shown}{more}")
    add("")

    # 9. Methodology and model details
    add("## 8. Methodology and Models")
    add("")
    add("Models used (no training or fine-tuning):")
    revisions = {
        "sentence-transformers/all-MiniLM-L6-v2": metadata.get("embedding_model_revision"),
        "cardiffnlp/twitter-roberta-base-sentiment-latest": metadata.get("sentiment_model_revision"),
    }
    for role, model_id, licence in MODEL_ATTRIBUTION:
        if role == "Subcategory discovery":
            model_id = metadata.get("llm_model_id") or model_id
        revision = revisions.get(model_id)
        add(f"- {role}: `{model_id}` ({licence})" + (f", revision `{revision}`" if revision else ""))
    add("")
    add("1. The input adapter combines the files and separates gold labels; preprocessing removes empty, invalid-date and duplicate rows.")
    add("2. Major category: MiniLM cosine similarity against `config/major_categories.yaml`, with `Other` as fallback.")
    add("3. Sentiment: CardiffNLP RoBERTa, run locally and independently of the category steps.")
    add("4. Subcategory: reviews are grouped by predicted major category and sent to Gemini in batches "
        "(review ID, text and predicted category only); batch results are merged by MiniLM similarity.")
    add("5. Priority: V/N/S/T heuristic per subcategory.")
    add("")
    add(f"- Gemini requests: {_fmt(metadata.get('api_request_count'))}, retries: {_fmt(metadata.get('api_retry_count'))}")
    durations = metadata.get("stage_durations_seconds") or {}
    if durations:
        add("- Stage durations (s): " + ", ".join(f"{key} {_fmt(value)}" for key, value in durations.items()))
    add(f"- Peak process RSS (MB): {_fmt(metadata.get('peak_process_rss_mb'), 1)}")
    add("- Representative reviews are the members with the highest major-category confidence.")
    add("")

    # 10. Limitations and interpretation
    add("## 9. Limitations and Interpretation")
    add("")
    add("- Predictions are model outputs, not ground truth.")
    add("- The sentiment model was trained on tweets; it measures tone, not how serious a defect is.")
    add("- Subcategories are discovered by an LLM for this run only; IDs are not stable across runs.")
    add("- Priority is a heuristic for ordering work, not a validated or accurate measure.")
    add("- Reviews in `UNASSIGNED` were not grouped and are not part of any priority score.")
    add("- Gemini free-tier content may be used by Google to improve its products; do not process confidential reviews.")
    add("")
    add("---")
    add("")
    add("Generated by App Review Intelligence.")
    add("")

    return str(_scrub_secrets("\n".join(lines)))


def generate_report(
    results: dict[str, Any],
    output_dir: str | Path,
    classified_reviews: list[dict[str, Any]] | None = None,
    clusters: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> str:
    """Write report.md (+ classified_reviews.csv and clusters.json when given). Returns the report path."""
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary.")
    if classified_reviews is not None and not isinstance(classified_reviews, list):
        raise TypeError("classified_reviews must be a list.")
    if run_metadata is not None and not isinstance(run_metadata, dict):
        raise TypeError("run_metadata must be a dictionary.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records = classified_reviews or []
    metadata = run_metadata or {}

    clusters_document = None
    if isinstance(clusters, dict):
        if "major_categories" in clusters:
            clusters_document = clusters
        else:
            clusters_document = build_clusters(
                clusters,
                records,
                results.get("priorities"),
                run_id=metadata.get("run_id"),
                llm_model_id=metadata.get("llm_model_id"),
            )

    report_path = _unique_report_path(output_path)
    report_path.write_text(_build_report(results, records, metadata, clusters_document), encoding="utf-8")

    if classified_reviews is not None:
        _write_csv(output_path / "classified_reviews.csv", classified_reviews)
    if clusters_document is not None:
        _write_json(output_path / "clusters.json", clusters_document)
    return str(report_path)


def write_run_metadata(output_dir: str | Path, run_metadata: dict[str, Any]) -> str:
    """Write run_metadata.json exclusively (never overwrites). Returns its path."""
    if not isinstance(run_metadata, dict):
        raise TypeError("run_metadata must be a dictionary.")
    path = Path(output_dir) / "run_metadata.json"
    _write_json(path, run_metadata)
    return str(path)
