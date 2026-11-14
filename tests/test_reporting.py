"""
Tests for reporting (Person 6). Report rules: Docs/09_CLI_AND_REPORT_SPECIFICATION.md.

Call shape used by these tests: generate_report(results, output_dir).
"""

from pathlib import Path

import pytest

from conftest import FAKE_API_KEY, get_component_function


GENERATE = ("app_review_intelligence.reporting.interface", "generate_report")


@pytest.fixture
def results():
    return {
        "major_category": {"accuracy": 0.90, "macro_f1": 0.88, "weighted_f1": 0.89},
        "sentiment": {"accuracy": 0.85, "macro_f1": 0.80, "weighted_f1": 0.84},
        # Subcategories are scored with clustering metrics, never "accuracy".
        "subcategory": {
            "ari": 0.41, "ami": 0.52, "nmi": 0.55, "v_measure": 0.55,
            "homogeneity": 0.60, "completeness": 0.51,
            "coverage": 0.97, "unassigned_ratio": 0.03,
        },
        "priorities": [],
    }


def _snapshot(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in Path(directory).rglob("*")
        if path.is_file()
    }


def test_reporting_creates_markdown_report(tmp_path, results):
    generate_report = get_component_function(*GENERATE)

    generate_report(results, str(tmp_path))

    markdown_files = list(Path(tmp_path).rglob("*.md"))

    assert len(markdown_files) > 0


def test_reporting_does_not_overwrite_existing_report(tmp_path, results):
    generate_report = get_component_function(*GENERATE)

    generate_report(results, str(tmp_path))
    first = _snapshot(tmp_path)

    generate_report(results, str(tmp_path))
    second = _snapshot(tmp_path)

    assert set(first).issubset(second)
    for path, content in first.items():
        assert second[path] == content, f"{path} was overwritten"


def test_report_states_priority_is_heuristic(tmp_path, results):
    generate_report = get_component_function(*GENERATE)

    generate_report(results, str(tmp_path))

    text = "\n".join(p.read_text(encoding="utf-8") for p in Path(tmp_path).rglob("*.md"))
    assert "heuristic" in text.lower()
    assert "priority accuracy" not in text.lower()


def test_report_never_contains_api_key(tmp_path, results, monkeypatch):
    generate_report = get_component_function(*GENERATE)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)

    generate_report(results, str(tmp_path))

    for content in _snapshot(tmp_path).values():
        assert FAKE_API_KEY.encode() not in content
