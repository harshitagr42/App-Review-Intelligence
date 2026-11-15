"""Public interface for reporting. No logic lives here.

Example:
    from app_review_intelligence.reporting.interface import generate_report

    report_path = generate_report(
        {"priorities": priority_results, "warnings": []},
        "outputs/run_20261115T101500Z_a1b2c3d4",
        classified_reviews=rows,                 # -> classified_reviews.csv
        clusters={"subcategories": [...], "assignments": [...]},  # -> clusters.json
        run_metadata=metadata,                   # shown in report.md
    )
"""

from app_review_intelligence.reporting.implementation import (
    generate_report as _generate_report,
    write_run_metadata as _write_run_metadata,
)


def generate_report(results, output_dir, classified_reviews=None, clusters=None, run_metadata=None):
    """
    Writes the run report and its artifacts. Existing files are never overwritten.

    Input:
        results: {"priorities": [...priority results...], "unassigned_review_ids": [...],
            "warnings": [...]}; extra keys are ignored.
        output_dir: the run folder.
        classified_reviews: one dict per accepted review (gold labels are dropped).
        clusters: the subcategory result {"subcategories", "assignments"}.
        run_metadata: run metadata shown in the report (not written here).

    Returns:
        Path of the Markdown report (report.md, or report_2.md if one exists).
    """
    return _generate_report(results, output_dir, classified_reviews, clusters, run_metadata)


def write_run_metadata(output_dir, run_metadata):
    """Writes run_metadata.json into output_dir (exclusive create). Returns its path."""
    return _write_run_metadata(output_dir, run_metadata)
