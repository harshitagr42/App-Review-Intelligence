
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)


def _validate_unique_ids(records, name):
    """Ensure every record has a unique, non-empty review ID."""
    if not isinstance(records, list):
        raise TypeError(f"{name} must be a list.")

    ids = []

    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"Every item in {name} must be a dictionary.")

        review_id = record.get("review_id")

        if review_id is None or str(review_id).strip() == "":
            raise ValueError(f"Missing review_id in {name}.")

        ids.append(str(review_id))

    duplicates = [
        review_id
        for review_id, count in Counter(ids).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate review IDs in {name}: {duplicates[:5]}"
        )

    return set(ids)


def _classification_metrics(actual, predicted):
    """Calculate standard classification metrics."""
    if not actual:
        return {
            "status": "not_available",
            "reason": "No labelled reviews available.",
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


def _subcategory_metrics(actual, predicted):
    """Evaluate discovered subcategories using clustering metrics."""
    if not actual:
        return {
            "status": "not_available",
            "reason": "No labelled reviews available.",
        }

    homogeneity, completeness, v_measure = (
        homogeneity_completeness_v_measure(actual, predicted)
    )

    return {
        "status": "available",
        "sample_count": len(actual),
        "adjusted_rand_index": float(
            adjusted_rand_score(actual, predicted)
        ),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(actual, predicted)
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(actual, predicted)
        ),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
    }


def evaluate_predictions(
    major_predictions,
    sentiment_results,
    subcategory_assignments,
    evaluation_labels,
):
    """
    Evaluate model predictions against gold labels.

    Gold labels are used only in this evaluation component.
    Predictions are matched using review_id, never list position.
    """

    # No evaluation labels means there is nothing to compare against.
    if not evaluation_labels:
        unavailable = {
            "status": "not_available",
            "reason": "No gold evaluation labels were provided.",
        }
        return {
            "major_category": unavailable,
            "sentiment": unavailable,
            "subcategory": unavailable,
            "evaluation_summary": {
                "status": "not_available",
                "labelled_review_count": 0,
            },
        }

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

    # Do not silently ignore missing or unexpected predictions.
    for name, prediction_ids in (
        ("major_predictions", major_ids),
        ("sentiment_results", sentiment_ids),
        ("subcategory_assignments", subcategory_ids),
    ):
        if prediction_ids != label_ids:
            missing = sorted(label_ids - prediction_ids)
            unexpected = sorted(prediction_ids - label_ids)

            raise ValueError(
                f"{name} review IDs do not match evaluation labels. "
                f"Missing: {missing[:5]}; "
                f"unexpected: {unexpected[:5]}."
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

    # Use the same deterministic ID order for every comparison.
    ordered_ids = sorted(label_ids)

    major_actual = []
    major_predicted = []
    sentiment_actual = []
    sentiment_predicted = []
    minor_actual = []
    subcategory_predicted = []

    for review_id in ordered_ids:
        gold = labels_by_id[review_id]

        major_actual.append(gold["gold_major_category"])
        major_predicted.append(
            major_by_id[review_id]["predicted_major_category"]
        )

        sentiment_actual.append(gold["gold_sentiment"])
        sentiment_predicted.append(
            sentiment_by_id[review_id]["sentiment"]
        )

        minor_actual.append(gold["gold_minor_category"])
        subcategory_predicted.append(
            subcategory_by_id[review_id]["subcategory_id"]
        )

    return {
        "major_category": _classification_metrics(
            major_actual, major_predicted
        ),
        "sentiment": _classification_metrics(
            sentiment_actual, sentiment_predicted
        ),
        "subcategory": _subcategory_metrics(
            minor_actual, subcategory_predicted
        ),
        "evaluation_summary": {
            "status": "available",
            "labelled_review_count": len(ordered_ids),
            "matching_method": "review_id",
        },
    }
