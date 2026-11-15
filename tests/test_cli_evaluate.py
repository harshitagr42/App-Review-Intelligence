
import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from app_review_intelligence.cli.implementation import app
from app_review_intelligence.adapters.input import interface as input_interface
from app_review_intelligence.evaluation import interface as evaluation_interface


runner = CliRunner()


@pytest.fixture
def run_directory(tmp_path):
    """Create a realistic run folder with saved prediction artifacts."""
    run_dir = tmp_path / "run_test123"
    run_dir.mkdir()

    pd.DataFrame([
        {
            "review_id": "1",
            "review_text": "The app crashes.",
            "predicted_major_category": "Performance",
            "sentiment": "Negative",
            "subcategory_id": "SUB_001",
        },
        {
            "review_id": "2",
            "review_text": "Great application.",
            "predicted_major_category": "Other",
            "sentiment": "Positive",
            "subcategory_id": "SUB_002",
        },
    ]).to_csv(run_dir / "classified_reviews.csv", index=False)

    (run_dir / "clusters.json").write_text(
        json.dumps({
            "subcategories": [],
            "assignments": [
                {"review_id": "1", "subcategory_id": "SUB_001"},
                {"review_id": "2", "subcategory_id": "SUB_002"},
            ],
        }),
        encoding="utf-8",
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps({"seed": 42, "model_ids": {}}),
        encoding="utf-8",
    )

    return run_dir


@pytest.fixture
def labelled_input(tmp_path):
    """Create a labelled input CSV."""
    path = tmp_path / "reviews.csv"

    pd.DataFrame([
        {
            "ID": "1",
            "Review": "The app crashes.",
            "Major Class": "Performance",
            "Minor Class": "Crashes",
            "Sentiment": "Negative",
        },
        {
            "ID": "2",
            "Review": "Great application.",
            "Major Class": "Other",
            "Minor Class": "General",
            "Sentiment": "Positive",
        },
    ]).to_csv(path, index=False)

    return path


def test_evaluate_runs_successfully(
    run_directory,
    labelled_input,
    monkeypatch,
):
    """Evaluate loads labels and calls the existing evaluator."""

    monkeypatch.setattr(
        input_interface,
        "load_and_prepare",
        lambda paths: (
            [],
            [
                {
                    "review_id": "1",
                    "gold_major_category": "Performance",
                    "gold_minor_category": "Crashes",
                    "gold_sentiment": "Negative",
                },
                {
                    "review_id": "2",
                    "gold_major_category": "Other",
                    "gold_minor_category": "General",
                    "gold_sentiment": "Positive",
                },
            ],
            {"rows_loaded": 2},
        ),
    )

    captured = {}

    def fake_evaluate_predictions(**kwargs):
        captured.update(kwargs)
        return {
            "evaluation_summary": {"status": "available"}
        }

    monkeypatch.setattr(
        evaluation_interface,
        "evaluate_predictions",
        fake_evaluate_predictions,
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--run-dir", str(run_directory),
            "--input", str(labelled_input),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluation completed successfully" in result.output
    assert len(captured["major_predictions"]) == 2
    assert len(captured["sentiment_results"]) == 2
    assert len(captured["evaluation_labels"]) == 2
    assert captured["output_dir"].is_dir()


@pytest.mark.parametrize(
    "missing_file",
    [
        "classified_reviews.csv",
        "clusters.json",
        "run_metadata.json",
    ],
)
def test_evaluate_rejects_missing_artifact(
    run_directory,
    labelled_input,
    missing_file,
):
    (run_directory / missing_file).unlink()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--run-dir", str(run_directory),
            "--input", str(labelled_input),
        ],
    )

    assert result.exit_code == 3
    assert "Required run artifact is missing" in result.output


def test_evaluate_rejects_input_without_gold_labels(
    run_directory,
    labelled_input,
    monkeypatch,
):
    monkeypatch.setattr(
        input_interface,
        "load_and_prepare",
        lambda paths: (
            [{"review_id": "1", "review_text": "Example"}],
            [],
            {"rows_loaded": 1},
        ),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--run-dir", str(run_directory),
            "--input", str(labelled_input),
        ],
    )

    assert result.exit_code == 1
    assert "ground-truth labels" in result.output
