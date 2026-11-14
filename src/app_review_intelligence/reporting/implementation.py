
import csv
import json
from pathlib import Path


def _format_value(value):
    """Format metric values for a readable Markdown report."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {_format_value(item)}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _metrics_section(title, metrics):
    """Convert one metric dictionary into a Markdown section."""
    lines = [f"## {title}", ""]

    if not isinstance(metrics, dict):
        lines.append("Metrics unavailable.")
        lines.append("")
        return lines

    if metrics.get("status") == "not_available":
        lines.append(metrics.get("reason", "Metrics unavailable."))
        lines.append("")
        return lines

    for key, value in metrics.items():
        if isinstance(value, dict) and key not in {
            "confusion_matrix"
        }:
            lines.append(f"### {key.replace('_', ' ').title()}")
            for subkey, subvalue in value.items():
                lines.append(
                    f"- **{subkey.replace('_', ' ').title()}:** "
                    f"{_format_value(subvalue)}"
                )
        elif key == "confusion_matrix":
            labels = value.get("labels", [])
            matrix = value.get("matrix", [])
            lines.append("### Confusion Matrix")
            if labels and matrix:
                lines.append("")
                lines.append(
                    "| Actual / Predicted | "
                    + " | ".join(str(label) for label in labels)
                    + " |"
                )
                lines.append(
                    "|---|" + "|".join("---:" for _ in labels) + "|"
                )
                for label, row in zip(labels, matrix):
                    lines.append(
                        f"| {label} | "
                        + " | ".join(str(item) for item in row)
                        + " |"
                    )
        else:
            lines.append(
                f"- **{key.replace('_', ' ').title()}:** "
                f"{_format_value(value)}"
            )

    lines.append("")
    return lines


def _write_json(path, data):
    """Write JSON without exposing unsupported Python objects."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)


def _write_csv(path, records):
    """Write a list of dictionaries to CSV."""
    if not records:
        return

    fieldnames = list(
        dict.fromkeys(
            key for record in records for key in record.keys()
        )
    )

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def _unique_report_path(output_dir):
    """Choose a report filename that never overwrites an existing one."""
    candidate = output_dir / "report.md"
    counter = 2

    while candidate.exists():
        candidate = output_dir / f"report_{counter}.md"
        counter += 1

    return candidate


def generate_report(
    results,
    output_dir,
    classified_reviews=None,
    clusters=None,
    run_metadata=None,
    logs=None,
):
    """
    Generate a Markdown report and optional run artifacts.

    output_dir must be the unique folder created for the current run.
    Existing report files are never overwritten.
    """
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = _unique_report_path(output_path)

    lines = [
        "# App Review Intelligence Report",
        "",
        "## Evaluation Overview",
        "",
    ]

    summary = results.get("evaluation_summary", {})
    if summary:
        lines.append(
            f"- **Status:** {summary.get('status', 'Not provided')}"
        )
        lines.append(
            "- **Labelled reviews:** "
            f"{summary.get('labelled_review_count', 'Not provided')}"
        )
        lines.append(
            "- **Matching method:** "
            f"{summary.get('matching_method', 'Not provided')}"
        )
        lines.append("")

    lines.extend(
        _metrics_section(
            "Major Category Evaluation",
            results.get("major_category", {}),
        )
    )

    lines.extend(
        _metrics_section(
            "Sentiment Evaluation",
            results.get("sentiment", {}),
        )
    )

    lines.extend(
        _metrics_section(
            "Subcategory Clustering Evaluation",
            results.get("subcategory", {}),
        )
    )

    lines.extend([
        "## Priority Analysis",
        "",
        "Priority scores are descriptive and do not represent "
        "prediction accuracy.",
        "",
    ])

    priorities = results.get("priorities", [])
    if priorities:
        lines.extend([
            "| Subcategory | Reviews | Negative proportion | "
            "Priority score | Priority level |",
            "|---|---:|---:|---:|---|",
        ])

        for item in priorities:
            lines.append(
                f"| {item.get('subcategory_name', 'Unknown')} "
                f"| {item.get('review_count', '')} "
                f"| {_format_value(item.get('negative_proportion', ''))} "
                f"| {_format_value(item.get('priority_score', ''))} "
                f"| {item.get('priority_level', '')} |"
            )
        lines.append("")
    else:
        lines.append("No priority results were provided.")
        lines.append("")

    lines.extend([
        "---",
        "",
        "Generated by App Review Intelligence.",
        "",
    ])

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # Optional artifacts supplied by the pipeline.
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
        log_text = (
            "\n".join(str(line) for line in logs)
            if isinstance(logs, list)
            else str(logs)
        )
        (output_path / "logs.txt").write_text(
            log_text,
            encoding="utf-8",
        )

    return str(report_path)
