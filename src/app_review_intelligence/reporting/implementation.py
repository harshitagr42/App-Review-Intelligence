
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


GOLD_FIELDS = {
    "gold_major_category",
    "gold_minor_category",
    "gold_sentiment",
}

DEFAULT_CSV_FIELDS = [
    "review_id",
    "review_text",
    "review_date",
    "predicted_major_category",
    "sentiment",
    "subcategory_id",
    "priority_level",
    "priority_score",
]


def _scrub_secrets(value: Any) -> Any:
    """Remove API keys from strings before writing report artifacts."""
    if isinstance(value, dict):
        return {
            key: _scrub_secrets(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_scrub_secrets(item) for item in value]

    if isinstance(value, tuple):
        return [_scrub_secrets(item) for item in value]

    if not isinstance(value, str):
        return value

    result = value

    # Remove the currently configured Gemini API key.
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        result = result.replace(api_key, "[REDACTED_API_KEY]")

    # Scrub common API-key assignment patterns.
    result = re.sub(
        r"(?i)(GEMINI_API_KEY\s*[:=]\s*)['\"]?[^'\"\s,;]+",
        r"\1[REDACTED]",
        result,
    )

    result = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*)['\"]?[^'\"\s,;]+",
        r"\1[REDACTED]",
        result,
    )

    return result


def _validate_json_value(value: Any, path: str = "root") -> None:
    """Reject non-finite numbers and unsupported JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite number found at {path}.")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be a string.")
            _validate_json_value(item, f"{path}.{key}")
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return

    raise TypeError(
        f"Unsupported JSON value at {path}: {type(value).__name__}"
    )


def _format_value(value: Any) -> str:
    """Format values for Markdown without exposing secrets."""
    value = _scrub_secrets(value)

    if value is None:
        return "Not available"

    if isinstance(value, float):
        if not math.isfinite(value):
            return "Not available"
        return f"{value:.4f}"

    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {_format_value(item)}"
            for key, item in value.items()
        )

    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)

    return str(value)


def _distribution(records: list[dict], possible_fields: tuple[str, ...]) -> dict:
    """Count values from the first available prediction field."""
    if not records:
        return {}

    field = next(
        (name for name in possible_fields
         if any(name in record for record in records)),
        None,
    )

    if field is None:
        return {}

    return dict(Counter(
        str(record.get(field, "Unknown"))
        for record in records
    ))


def _metrics_section(title: str, metrics: dict) -> list[str]:
    """Render a metric dictionary as Markdown."""
    lines = [f"### {title}", ""]

    if not isinstance(metrics, dict) or not metrics:
        return lines + ["Metrics unavailable.", ""]

    if metrics.get("status") == "not_available":
        return lines + [
            metrics.get("reason", "Metrics unavailable."),
            "",
        ]

    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"**{key.replace('_', ' ').title()}**")
            for subkey, subvalue in value.items():
                lines.append(
                    f"- {subkey.replace('_', ' ').title()}: "
                    f"{_format_value(subvalue)}"
                )
        elif key != "status":
            lines.append(
                f"- **{key.replace('_', ' ').title()}:** "
                f"{_format_value(value)}"
            )

    lines.append("")
    return lines


def _unique_report_path(output_dir: Path) -> Path:
    """Reserve a unique Markdown report filename."""
    counter = 1

    while True:
        name = "report.md" if counter == 1 else f"report_{counter}.md"
        candidate = output_dir / name

        try:
            with candidate.open("x", encoding="utf-8"):
                pass
            return candidate
        except FileExistsError:
            counter += 1


def _write_json(path: Path, data: Any) -> None:
    """Validate and exclusively write strict JSON."""
    cleaned = _scrub_secrets(data)
    _validate_json_value(cleaned)

    serialized = json.dumps(
        cleaned,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    with path.open("x", encoding="utf-8") as file:
        file.write(serialized)



def _prepare_csv_records(
    records: list[dict],
) -> tuple[list[dict], list[str]]:
    """Validate classified-review records and remove gold-label columns."""

    if not isinstance(records, list):
        raise TypeError("classified_reviews must be a list.")

    required_fields = {"review_id", "review_text"}

    allowed_fields = {
        "review_id",
        "review_text",
        "review_date",
        "predicted_major_category",
        "major_category_confidence",
        "sentiment",
        "sentiment_confidence",
        "subcategory_id",
        "subcategory_name",
        "priority_level",
        "priority_score",
    }

    cleaned_records = []
    seen_ids = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"CSV record {index} must be a dictionary."
            )

        # Remove gold labels before exporting.
        cleaned = {
            key: _scrub_secrets(value)
            for key, value in record.items()
            if key not in GOLD_FIELDS
        }

        if not required_fields.issubset(cleaned):
            missing = required_fields - set(cleaned)
            raise ValueError(
                f"CSV record {index} is missing required fields: "
                f"{sorted(missing)}"
            )

        unexpected = set(cleaned) - allowed_fields

        if unexpected:
            raise ValueError(
                f"CSV record {index} contains unsupported fields: "
                f"{sorted(unexpected)}"
            )

        review_id = cleaned["review_id"]
        review_text = cleaned["review_text"]

        if review_id is None or not str(review_id).strip():
            raise ValueError(
                f"CSV record {index} has an empty review_id."
            )

        if not isinstance(review_text, str) or not review_text.strip():
            raise ValueError(
                f"CSV record {index} has invalid review_text."
            )

        normalized_id = str(review_id).strip()

        if normalized_id in seen_ids:
            raise ValueError(
                f"Duplicate review_id found in CSV records: {normalized_id}"
            )

        seen_ids.add(normalized_id)
        cleaned_records.append(cleaned)

    if cleaned_records:
        fieldnames = list(dict.fromkeys(
            key
            for record in cleaned_records
            for key in record
        ))
    else:
        fieldnames = list(DEFAULT_CSV_FIELDS)

    return cleaned_records, fieldnames



def _write_csv(path: Path, records: list[dict]) -> None:
    """Validate and exclusively write CSV, including empty datasets."""
    cleaned_records, fieldnames = _prepare_csv_records(records)

    with path.open(
        "x",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(cleaned_records)


def _write_logs(path: Path, logs: Any) -> None:
    """Write logs after removing API keys."""
    if isinstance(logs, list):
        text = "\n".join(str(line) for line in logs)
    else:
        text = str(logs)

    text = _scrub_secrets(text)

    with path.open("x", encoding="utf-8") as file:
        file.write(text)


def _build_report(
    results: dict,
    records: list[dict],
    metadata: dict,
    clusters: Any,
) -> str:
    """Build the ten-section run report."""
    major_distribution = _distribution(
        records,
        ("predicted_major_category", "major_category"),
    )

    sentiment_distribution = _distribution(
        records,
        ("sentiment", "predicted_sentiment"),
    )

    unassigned = [
        record for record in records
        if str(record.get("subcategory_id", "")).strip().casefold()
        == "unassigned"
    ]

    summary = results.get("evaluation_summary", {})
    subcategory = results.get("subcategory", {})
    priorities = (
        results.get("priorities")
        or results.get("priority_analysis", {}).get("top_subcategories", [])
    )

    lines = [
        "# App Review Intelligence — Run Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Run status: {summary.get('status', 'Not provided')}",
        f"- Reviews processed: {len(records)}",
        f"- Evaluation status: {summary.get('status', 'Not provided')}",
        "",
        "## 2. Input Summary",
        "",
        f"- Input files: {_format_value(metadata.get('input_files'))}",
        f"- Input file hashes: {_format_value(metadata.get('input_file_metadata'))}",
        f"- Input rows: {_format_value(metadata.get('input_row_count'))}",
        f"- Accepted rows: {_format_value(metadata.get('accepted_row_count'))}",
        f"- Excluded rows: {_format_value(metadata.get('excluded_row_count'))}",
        "",
        "## 3. Major-Category Distribution",
        "",
    ]

    if major_distribution:
        for category, count in sorted(major_distribution.items()):
            lines.append(f"- {category}: {count}")
    else:
        lines.append("Major-category distribution unavailable.")

    lines.extend([
        "",
        "## 4. Sentiment Distribution",
        "",
    ])

    if sentiment_distribution:
        for sentiment, count in sorted(sentiment_distribution.items()):
            lines.append(f"- {sentiment}: {count}")
    else:
        lines.append("Sentiment distribution unavailable.")

    lines.extend([
        "",
        "## 5. Unassigned Reviews Summary",
        "",
        f"- Unassigned reviews in supplied records: {len(unassigned)}",
        f"- Unassigned ratio: {_format_value(subcategory.get('unassigned_ratio'))}",
        f"- Coverage: {_format_value(subcategory.get('coverage'))}",
        "",
        "## 6. Evaluation Results",
        "",
    ])

    lines.extend(_metrics_section(
        "Major Category Classification",
        results.get("major_category", {}),
    ))
    lines.extend(_metrics_section(
        "Sentiment Classification",
        results.get("sentiment", {}),
    ))
    lines.extend(_metrics_section(
        "Subcategory Clustering",
        subcategory,
    ))

    lines.extend([
        "## 7. Priority Analysis",
        "",
        "Priority scores are heuristic, descriptive indicators. "
        "They are not ground-truth predictions and must not be treated "
        "as measures of predictive performance."
        "",
    ])

    if priorities:
        lines.extend([
            "| Subcategory | Review count | Priority score | Priority level |",
            "|---|---:|---:|---|",
        ])

        for item in priorities[:10]:
            lines.append(
                f"| {_format_value(item.get('subcategory_name'))} "
                f"| {_format_value(item.get('review_count'))} "
                f"| {_format_value(item.get('priority_score'))} "
                f"| {_format_value(item.get('priority_level'))} |"
            )
    else:
        lines.append("No priority results were provided.")

    lines.extend([
        "",
        "## 8. Methodology and Models",
        "",
        f"- Models: {_format_value(metadata.get('model_ids', {}))}",
        f"- Model revisions: {_format_value(metadata.get('model_revisions', {}))}",
        f"- Random seed: {_format_value(metadata.get('seed'))}",
        f"- Runtime (seconds): {_format_value(metadata.get('runtime_seconds'))}",
        f"- Memory (MB): {_format_value(metadata.get('memory_mb'))}",
        "",
        "The pipeline uses the configured classification, sentiment, "
        "subcategory discovery, and priority components.",
        "",
        "## 9. Clusters and Run Artifacts",
        "",
        f"- Discovered subcategory count: "
        f"{_format_value(subcategory.get('discovered_subcategory_count'))}",
        f"- Cluster artifact supplied: {'Yes' if clusters is not None else 'No'}",
        "- Detailed classifications and cluster data are stored in their "
        "respective run artifacts when supplied.",
        "",
        "## 10. Limitations",
        "",
        "- Subcategory discovery is evaluated using clustering metrics, "
        "not classification accuracy.",
        "- Priority scores are heuristic and descriptive, not predictive "
        "accuracy measurements.",
        "- Evaluation metrics depend on the availability and quality of "
        "gold labels.",
        "- Missing or unknown labels and excluded records should be reviewed "
        "before interpreting evaluation results.",
        "",
        "---",
        "",
        "Generated by App Review Intelligence.",
        "",
    ])

    return str(_scrub_secrets("\n".join(lines)))


def generate_report(
    results: dict[str, Any],
    output_dir: str | Path,
    classified_reviews: list[dict[str, Any]] | None = None,
    clusters: dict[str, Any] | list[Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
    logs: Any = None,
) -> str:
    """
    Generate a ten-section Markdown run report and optional artifacts.

    The caller is responsible for supplying a unique run directory.
    Existing artifact files are never overwritten.
    """
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary.")

    if classified_reviews is not None and not isinstance(
        classified_reviews, list
    ):
        raise TypeError("classified_reviews must be a list.")

    if run_metadata is not None and not isinstance(run_metadata, dict):
        raise TypeError("run_metadata must be a dictionary.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = classified_reviews or []
    metadata = run_metadata or {}

    report_text = _build_report(
        results=results,
        records=records,
        metadata=metadata,
        clusters=clusters,
    )

    report_path = _unique_report_path(output_path)
    report_path.write_text(report_text, encoding="utf-8")

    # These fixed artifact names are exclusive: no silent overwrites.
    if classified_reviews is not None:
        _write_csv(
            output_path / "classified_reviews.csv",
            classified_reviews,
        )

    if clusters is not None:
        _write_json(output_path / "clusters.json", clusters)

    if run_metadata is not None:
        _write_json(
            output_path / "run_metadata.json",
            run_metadata,
        )

    if logs is not None:
        _write_logs(output_path / "logs.txt", logs)

    return str(report_path)
