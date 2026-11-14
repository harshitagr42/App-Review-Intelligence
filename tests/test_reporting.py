
from pathlib import Path

from conftest import get_component_function


def test_reporting_creates_markdown_report(tmp_path):
    generate_report = get_component_function(
        "app_review_intelligence.reporting.interface",
        "generate_report",
    )

    results = {
        "major_category": {"accuracy": 0.90},
        "sentiment": {"accuracy": 0.85},
        "subcategory": {
            "adjusted_rand_index": 0.40,
            "adjusted_mutual_info": 0.55,
            "normalized_mutual_info": 0.56,
            "v_measure": 0.56,
        },
        "priorities": [],
    }

    generate_report(results, str(tmp_path))

    markdown_files = list(Path(tmp_path).glob("*.md"))

    assert len(markdown_files) > 0


def test_reporting_does_not_overwrite_existing_report(tmp_path):
    generate_report = get_component_function(
        "app_review_intelligence.reporting.interface",
        "generate_report",
    )

    results = {
        "major_category": {"accuracy": 0.90},
        "sentiment": {"accuracy": 0.85},
        "subcategory": {
            "adjusted_rand_index": 0.40,
            "adjusted_mutual_info": 0.55,
            "normalized_mutual_info": 0.56,
            "v_measure": 0.56,
        },
        "priorities": [],
    }

    generate_report(results, str(tmp_path))
    first_files = set(tmp_path.iterdir())

    generate_report(results, str(tmp_path))
    second_files = set(tmp_path.iterdir())

    assert first_files.issubset(second_files)
