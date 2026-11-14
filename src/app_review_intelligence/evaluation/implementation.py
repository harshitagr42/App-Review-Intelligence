from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

UNASSIGNED = "UNASSIGNED"


def _validate_unique_ids(records: list[dict], name: str) -> set[str]:
    """Validate record structure and return normalized review IDs."""
    if not isinstance(records, list):
        raise TypeError(f"{name} must be a list.")

    ids = []

    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"Every item in {name} must be a dictionary.")

        review_id = record.get("review_id")

        if review_id is None or not str(review_id).strip():
            raise ValueError(f"Missing review_id in {name}.")

        ids.append(str(review_id))

    duplicates = [
        item for item, count in Counter(ids).items() if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate review IDs in {name}: {duplicates[:5]}"
        )

    return set(ids)


def _safe_float(value: Any) -> float | None:
    """Convert numeric values to strict-JSON-safe floats."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def _classification_metrics(
    actual: list[str],
    predicted: list[str],
) -> dict[str, Any]:
    """Calculate classification metrics, including flat F1 fields."""
    if not actual:
        return {
            "status": "not_available",
            "reason": "No eligible labelled reviews available.",
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    labels = sorted(set(actual) | set(predicted))

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "status": "available",
        "sample_count": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1_score": float(weighted_f1),
        },
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(
                labels, precision, recall, f1, support
            )
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(
                actual, predicted, labels=labels
            ).tolist(),
        },
    }


def _subcategory_metrics(
    actual: list[str],
    predicted: list[str],
    total_count: int,
    assigned_count: int,
) -> dict[str, Any]:
    """Calculate clustering metrics without treating UNASSIGNED as a cluster."""
    coverage = assigned_count / total_count if total_count else 0.0
    unassigned_ratio = (
        (total_count - assigned_count) / total_count
        if total_count else 0.0
    )

    result = {
        "status": "not_available",
        "sample_count": assigned_count,
        "total_review_count": total_count,
        "coverage": float(coverage),
        "unassigned_ratio": float(unassigned_ratio),
        "unassigned_count": total_count - assigned_count,
        "assigned_count": assigned_count,
        "discovered_subcategory_count": len(set(predicted)),
        "represented_gold_minor_class_count": len(set(actual)),
    }

    if not actual:
        result["reason"] = "No assigned reviews available for clustering evaluation."
        return result

    homogeneity, completeness, v_measure = (
        homogeneity_completeness_v_measure(actual, predicted)
    )

    ari = adjusted_rand_score(actual, predicted)
    ami = adjusted_mutual_info_score(actual, predicted)
    nmi = normalized_mutual_info_score(actual, predicted)

    result.update({
        "status": "available",
        "ari": float(ari),
        "ami": float(ami),
        "nmi": float(nmi),
        "v_measure": float(v_measure),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),

        # Descriptive aliases retained for compatibility.
        "adjusted_rand_index": float(ari),
        "adjusted_mutual_information": float(ami),
        "normalized_mutual_information": float(nmi),
    })

    return result


def _per_major_breakdown(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Provide classification and subcategory coverage by gold major class."""
    breakdown = {}

    major_classes = sorted({
        record["gold_major_category"]
        for record in records
        if record.get("gold_major_category") not in (None, "")
    })

    for major_class in major_classes:
        subset = [
            record for record in records
            if record.get("gold_major_category") == major_class
        ]

        major_actual = [
            item["gold_major_category"] for item in subset
        ]
        major_predicted = [
            item["predicted_major_category"] for item in subset
        ]

        sentiment_actual = [
            item["gold_sentiment"].strip().casefold()
            for item in subset
        ]
        sentiment_predicted = [
            item["predicted_sentiment"].strip().casefold()
            for item in subset
        ]

        assigned = [
            item for item in subset
            if item["subcategory_id"].strip().casefold()
            != UNASSIGNED.casefold()
        ]

        breakdown[major_class] = {
            "review_count": len(subset),
            "major_category_accuracy": float(
                accuracy_score(major_actual, major_predicted)
            ) if subset else None,
            "sentiment_accuracy": float(
                accuracy_score(sentiment_actual, sentiment_predicted)
            ) if subset else None,
            "subcategory_coverage": (
                len(assigned) / len(subset) if subset else 0.0
            ),
            "assigned_review_count": len(assigned),
            "unassigned_review_count": len(subset) - len(assigned),
        }

    return breakdown


def _priority_summary(
    priority_results: list[dict] | None,
) -> dict[str, Any]:
    """Summarize priorities descriptively, never as predictive accuracy."""
    if not priority_results:
        return {
            "status": "not_available",
            "top_subcategories": [],
            "limitation": (
                "Priority scores are heuristic and descriptive. "
                "They are not ground-truth predictions and are not "
                "evaluated using accuracy, precision, recall, or F1."
            ),
        }

    ranked = sorted(
        priority_results,
        key=lambda item: _safe_float(item.get("priority_score")) or 0.0,
        reverse=True,
    )

    return {
        "status": "available",
        "top_subcategories": ranked[:10],
        "total_subcategory_count": len(priority_results),
        "limitation": (
            "Priority scores are heuristic and descriptive. "
            "They do not represent predictive accuracy or ground truth."
        ),
    }


def _build_metadata(
    run_metadata: dict[str, Any] | None,
    input_file: str | Path | None,
    total_rows: int,
    evaluated_rows: int,
    seed: int | None,
    model_ids: dict[str, Any] | None,
    runtime_seconds: float | None,
    memory_mb: float | None,
) -> dict[str, Any]:
    """Build reproducibility metadata and hash the input when available."""
    metadata = dict(run_metadata or {})

    metadata["total_input_rows"] = total_rows
    metadata["evaluated_rows"] = evaluated_rows
    metadata["excluded_rows"] = total_rows - evaluated_rows

    if seed is not None:
        metadata["seed"] = seed

    if model_ids is not None:
        metadata["model_ids"] = model_ids

    if runtime_seconds is not None:
        metadata["runtime_seconds"] = _safe_float(runtime_seconds)

    if memory_mb is not None:
        metadata["memory_mb"] = _safe_float(memory_mb)

    if input_file is not None:
        path = Path(input_file)
        metadata["input_file_name"] = path.name

        if path.is_file():
            digest = hashlib.sha256()

            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)

            metadata["input_sha256"] = digest.hexdigest()
        else:
            metadata["input_sha256"] = None

    return metadata


def _write_evaluation_files(
    result: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Write evaluation_report.md and strict evaluation.json."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / "evaluation.json"
    report_path = directory / "evaluation_report.md"

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    major = result["major_category"]
    sentiment = result["sentiment"]
    subcategory = result["subcategory"]
    summary = result["evaluation_summary"]
    priority = result["priority_analysis"]

    lines = [
        "# Evaluation Report",
        "",
        "## Evaluation Summary",
        f"- Status: {summary['status']}",
        f"- Labelled reviews: {summary['labelled_review_count']}",
        f"- Evaluated reviews: {summary['evaluated_review_count']}",
        "",
        "## Major Category",
        f"- Accuracy: {major.get('accuracy')}",
        f"- Macro F1: {major.get('macro_f1')}",
        f"- Weighted F1: {major.get('weighted_f1')}",
        "",
        "## Sentiment",
        f"- Accuracy: {sentiment.get('accuracy')}",
        f"- Macro F1: {sentiment.get('macro_f1')}",
        f"- Weighted F1: {sentiment.get('weighted_f1')}",
        "",
        "## Subcategory Clustering",
        f"- ARI: {subcategory.get('ari')}",
        f"- AMI: {subcategory.get('ami')}",
        f"- NMI: {subcategory.get('nmi')}",
        f"- V-measure: {subcategory.get('v_measure')}",
        f"- Coverage: {subcategory.get('coverage')}",
        f"- Unassigned ratio: {subcategory.get('unassigned_ratio')}",
        f"- Discovered subcategories: {subcategory.get('discovered_subcategory_count')}",
        f"- Represented gold minor classes: {subcategory.get('represented_gold_minor_class_count')}",
        "",
        "## Priority Analysis",
        priority["limitation"],
        "",
        "## Exclusions and Unknown Labels",
        json.dumps(result["exclusions_and_unknown_labels"], indent=2),
        "",
        "## Per-Major-Class Breakdown",
        json.dumps(result["per_major_class"], indent=2),
        "",
        "## Run Metadata",
        json.dumps(result["run_metadata"], indent=2),
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_predictions(
    major_predictions: list[dict[str, Any]],
    sentiment_results: list[dict[str, Any]],
    subcategory_assignments: list[dict[str, Any]],
    evaluation_labels: list[dict[str, Any]],
    *,
    output_dir: str | Path | None = None,
    run_metadata: dict[str, Any] | None = None,
    priority_results: list[dict[str, Any]] | None = None,
    excluded_records: list[dict[str, Any]] | None = None,
    input_file: str | Path | None = None,
    seed: int | None = 42,
    model_ids: dict[str, Any] | None = None,
    runtime_seconds: float | None = None,
    memory_mb: float | None = None,
    known_major_categories: set[str] | None = None,
    known_sentiments: set[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate predictions against gold labels.

    Gold labels are accessed only in this evaluation function.
    All prediction matching is performed by review_id.

    If prediction IDs do not match the label IDs, the evaluator returns
    an incomplete result using only IDs present in every input.
    """

    start_time = time.perf_counter()

    if not isinstance(evaluation_labels, list):
        raise TypeError("evaluation_labels must be a list.")

    label_ids = _validate_unique_ids(
        evaluation_labels, "evaluation_labels"
    )
    major_ids = _validate_unique_ids(
        major_predictions, "major_predictions"
    )
    sentiment_ids = _validate_unique_ids(
        sentiment_results, "sentiment_results"
    )
    subcategory_ids = _validate_unique_ids(
        subcategory_assignments, "subcategory_assignments"
    )

    prediction_sets = {
        "major_predictions": major_ids,
        "sentiment_results": sentiment_ids,
        "subcategory_assignments": subcategory_ids,
    }

    mismatches = {}

    for name, ids in prediction_sets.items():
        missing = sorted(label_ids - ids)
        unexpected = sorted(ids - label_ids)

        if missing or unexpected:
            mismatches[name] = {
                "missing_review_ids": missing,
                "unexpected_review_ids": unexpected,
            }

    common_ids = (
        label_ids
        & major_ids
        & sentiment_ids
        & subcategory_ids
    )

    labels_by_id = {
        str(item["review_id"]): item
        for item in evaluation_labels
    }
    major_by_id = {
        str(item["review_id"]): item
        for item in major_predictions
    }
    sentiment_by_id = {
        str(item["review_id"]): item
        for item in sentiment_results
    }
    subcategory_by_id = {
        str(item["review_id"]): item
        for item in subcategory_assignments
    }

    records = []

    for review_id in sorted(common_ids):
        gold = labels_by_id[review_id]
        major = major_by_id[review_id]
        sentiment = sentiment_by_id[review_id]
        subcategory = subcategory_by_id[review_id]

        records.append({
            "review_id": review_id,
            "gold_major_category": gold.get("gold_major_category"),
            "predicted_major_category": major.get(
                "predicted_major_category"
            ),
            "gold_sentiment": gold.get("gold_sentiment"),
            "predicted_sentiment": sentiment.get("sentiment"),
            "gold_minor_category": gold.get("gold_minor_category"),
            "subcategory_id": str(
                subcategory.get("subcategory_id", UNASSIGNED)
            ),
        })

    # Unknown and excluded labels are reported separately.
    unknown_labels = {
        "major_category": [],
        "sentiment": [],
    }

    for record in records:
        gold_major = record["gold_major_category"]
        gold_sentiment = record["gold_sentiment"]

        if (
            known_major_categories is not None
            and gold_major not in known_major_categories
        ):
            unknown_labels["major_category"].append({
                "review_id": record["review_id"],
                "label": gold_major,
            })

        if (
            known_sentiments is not None
            and str(gold_sentiment).casefold()
            not in {str(item).casefold() for item in known_sentiments}
        ):
            unknown_labels["sentiment"].append({
                "review_id": record["review_id"],
                "label": gold_sentiment,
            })

    valid_major_records = [
        item for item in records
        if item["gold_major_category"] not in (None, "")
        and item["predicted_major_category"] not in (None, "")
    ]

    valid_sentiment_records = [
        item for item in records
        if item["gold_sentiment"] not in (None, "")
        and item["predicted_sentiment"] not in (None, "")
    ]

    major_actual = [
        item["gold_major_category"] for item in valid_major_records
    ]
    major_predicted = [
        item["predicted_major_category"] for item in valid_major_records
    ]

    # Normalize sentiment independently, preserving other label comparisons.
    sentiment_actual = [
        str(item["gold_sentiment"]).strip().casefold()
        for item in valid_sentiment_records
    ]
    sentiment_predicted = [
        str(item["predicted_sentiment"]).strip().casefold()
        for item in valid_sentiment_records
    ]

    assigned_records = [
        item for item in records
        if item["subcategory_id"].strip().casefold()
        != UNASSIGNED.casefold()
        and item["gold_minor_category"] not in (None, "")
    ]

    minor_actual = [
        item["gold_minor_category"] for item in assigned_records
    ]
    subcategory_predicted = [
        item["subcategory_id"] for item in assigned_records
    ]

    subcategory_result = _subcategory_metrics(
        minor_actual,
        subcategory_predicted,
        total_count=len(records),
        assigned_count=len(assigned_records),
    )

    # If IDs are mismatched, the result is explicitly incomplete.
    status = "incomplete" if mismatches else "available"

    if not records:
        status = "incomplete"

    excluded = list(excluded_records or [])

    missing_gold_fields = [
        {
            "review_id": item["review_id"],
            "missing_fields": [
                field for field in (
                    "gold_major_category",
                    "gold_minor_category",
                    "gold_sentiment",
                )
                if item.get(field) in (None, "")
            ],
        }
        for item in records
        if any(
            item.get(field) in (None, "")
            for field in (
                "gold_major_category",
                "gold_minor_category",
                "gold_sentiment",
            )
        )
    ]

    summary = {
        "status": status,
        "labelled_review_count": len(label_ids),
        "evaluated_review_count": len(records),
        "matching_method": "review_id",
        "mismatched_ids": mismatches,
        "mismatch_count": sum(
            len(item["missing_review_ids"])
            + len(item["unexpected_review_ids"])
            for item in mismatches.values()
        ),
    }

    result = {
        "major_category": _classification_metrics(
            major_actual, major_predicted
        ),
        "sentiment": _classification_metrics(
            sentiment_actual, sentiment_predicted
        ),
        "subcategory": subcategory_result,
        "evaluation_summary": summary,
        "per_major_class": _per_major_breakdown(records),
        "exclusions_and_unknown_labels": {
            "excluded_record_count": len(excluded),
            "excluded_records": excluded,
            "missing_gold_fields": missing_gold_fields,
            "unknown_labels": unknown_labels,
        },
        "priority_analysis": _priority_summary(priority_results),
    }

    elapsed = time.perf_counter() - start_time

    if runtime_seconds is None:
        runtime_seconds = elapsed

    metadata = _build_metadata(
        run_metadata=run_metadata,
        input_file=input_file,
        total_rows=len(label_ids),
        evaluated_rows=len(records),
        seed=seed,
        model_ids=model_ids,
        runtime_seconds=runtime_seconds,
        memory_mb=memory_mb,
    )

    result["run_metadata"] = metadata

    if output_dir is not None:
        _write_evaluation_files(result, output_dir)

    return result