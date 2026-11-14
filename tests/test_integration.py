"""
End-to-end integration test (Person 6, Docs/11 section 1): the whole pipeline
runs on a small synthetic file with the real local models and a FAKE Gemini
client (conftest.fake_gemini). The real Gemini API is never called.
"""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import FAKE_API_KEY, get_component_function


pytestmark = pytest.mark.local_model


@pytest.fixture(autouse=True)
def no_real_gemini_check(monkeypatch):
    """The pre-flight Gemini check would hit the network; replace it with a no-op."""
    from app_review_intelligence.components.subcategory import interface

    monkeypatch.setattr(interface, "check_gemini_access", lambda *args, **kwargs: None)

RUN_PIPELINE = ("app_review_intelligence.pipeline.interface", "run_pipeline")

ROWS = [
    ("RVW001", "The app crashes every time I open it.", "2026-07-01", "Performance and Stability", "App Crashes", "negative"),
    ("RVW002", "It freezes and crashes after the update.", "2026-07-02", "Performance and Stability", "App Crashes", "negative"),
    ("RVW003", "I was charged twice for my subscription.", "2026-07-03", "Billing and Payments", "Duplicate Charges", "negative"),
    ("RVW004", "My refund never arrived, terrible.", "2026-07-04", "Billing and Payments", "Refund Delays", "negative"),
    ("RVW005", "Great app, love the clean design!", "2026-07-05", "User Interface and Experience", "Design Praise", "positive"),
    ("RVW006", "I cannot log in, the OTP never arrives.", "2026-07-06", "Authentication and Account Access", "OTP Issues", "negative"),
    ("RVW007", "", "2026-07-07", "Other", "Empty", "neutral"),                                   # empty review
    ("RVW008", "The app crashes every time I open it.", "2026-07-01", "Performance and Stability", "App Crashes", "negative"),  # duplicate
]


@pytest.fixture
def labelled_csv(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame(
        ROWS, columns=["ID", "Review", "Date", "Major Class", "Minor Class", "Sentiment"]
    ).to_csv(path, index=False)
    return path


def _model_records(path):
    """Accepted records with the category the real classifier predicts (the fake Gemini needs it)."""
    load_and_prepare = get_component_function("app_review_intelligence.adapters.input.interface", "load_and_prepare")
    classify = get_component_function("app_review_intelligence.components.major_category.interface", "classify")
    records, _, _ = load_and_prepare([str(path)])
    predicted = {item["review_id"]: item["predicted_major_category"] for item in classify(records)}
    return [
        {"review_id": r["review_id"], "review_text": r["review_text"], "predicted_major_category": predicted[r["review_id"]]}
        for r in records
    ]


def _all_text(folder):
    return "\n".join(p.read_text(encoding="utf-8") for p in Path(folder).iterdir() if p.is_file())


def test_full_run_writes_every_output(labelled_csv, tmp_path, fake_gemini):
    run_pipeline = get_component_function(*RUN_PIPELINE)
    fake_gemini(_model_records(labelled_csv))

    result = run_pipeline([str(labelled_csv)], {
        "allow_cloud_processing": True, "output_dir": str(tmp_path / "out"),
    })

    folder = Path(result["run_folder"])
    assert folder.name.startswith("run_")
    assert {p.name for p in folder.iterdir()} == {
        "report.md", "classified_reviews.csv", "clusters.json", "run_metadata.json", "logs.txt",
    }

    with open(folder / "classified_reviews.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["review_id"] for row in rows] == ["RVW001", "RVW002", "RVW003", "RVW004", "RVW005", "RVW006"]  # 7 empty, 8 duplicate
    assert not {"Major Class", "Minor Class", "gold_major_category", "gold_sentiment"} & set(rows[0])
    assert all(row["sentiment"] in {"Positive", "Negative", "Neutral"} for row in rows)

    clusters = json.loads((folder / "clusters.json").read_text(encoding="utf-8"))
    assert clusters["schema_version"] == "1.0" and clusters["run_id"] == folder.name
    members = [rid for group in clusters["major_categories"] for sub in group["subcategories"]
               for rid in sub["member_review_ids"]]
    assert sorted(members + clusters["unassigned_review_ids"]) == sorted(row["review_id"] for row in rows)

    metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert (metadata["rows_read"], metadata["rows_accepted"]) == (8, 6)
    assert metadata["rows_skipped_empty"] == 1 and metadata["duplicate_review_count"] == 1
    assert metadata["api_request_count"] >= 1
    assert metadata["error"] is None

    report = (folder / "report.md").read_text(encoding="utf-8")
    for heading in ("Executive Summary", "Input Summary", "Major-Category Distribution", "Priority Ranking",
                    "Detailed Subcategory Analysis", "Sentiment Distribution", "Unassigned Reviews",
                    "Methodology and Models", "Limitations"):
        assert heading in report
    assert "heuristic" in report.lower()

    logs = (folder / "logs.txt").read_text(encoding="utf-8")
    assert "INFO" in logs and "Stage subcategory" in logs
    assert "crashes every time" not in logs          # no review text in logs
    assert FAKE_API_KEY not in _all_text(folder)      # never the API key


def test_gemini_receives_only_id_text_and_category(labelled_csv, tmp_path, fake_gemini):
    run_pipeline = get_component_function(*RUN_PIPELINE)
    state = fake_gemini(_model_records(labelled_csv))

    run_pipeline([str(labelled_csv)], {"allow_cloud_processing": True, "output_dir": str(tmp_path / "out")})

    prompts = "\n".join(state["calls"])
    for gold in ("App Crashes", "Duplicate Charges", "Refund Delays", "OTP Issues", "2026-07-0"):
        assert gold not in prompts


def test_two_runs_create_different_folders(labelled_csv, tmp_path, fake_gemini):
    run_pipeline = get_component_function(*RUN_PIPELINE)
    fake_gemini(_model_records(labelled_csv))
    config = {"allow_cloud_processing": True, "output_dir": str(tmp_path / "out")}

    first = run_pipeline([str(labelled_csv)], config)
    first_files = {p.name: p.read_bytes() for p in Path(first["run_folder"]).iterdir()}
    second = run_pipeline([str(labelled_csv)], config)

    assert first["run_folder"] != second["run_folder"]
    assert {p.name: p.read_bytes() for p in Path(first["run_folder"]).iterdir()} == first_files


def test_evaluate_mode_writes_evaluation_outputs(labelled_csv, tmp_path, fake_gemini):
    run_pipeline = get_component_function(*RUN_PIPELINE)
    fake_gemini(_model_records(labelled_csv))

    result = run_pipeline([str(labelled_csv)], {
        "mode": "evaluate", "allow_cloud_processing": True, "output_dir": str(tmp_path / "out"),
    })

    folder = Path(result["run_folder"])
    assert folder.name.startswith("eval_")
    assert {p.name for p in folder.iterdir()} == {
        "evaluation_report.md", "evaluation.json", "run_metadata.json", "logs.txt",
    }
    evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["evaluation_summary"]["evaluated_review_count"] == 6
    assert "accuracy" not in evaluation["subcategory"]
    report = (folder / "evaluation_report.md").read_text(encoding="utf-8")
    assert "Confusion matrix" in report and "ARI" in report


def test_run_refuses_without_consent(labelled_csv, tmp_path):
    run_pipeline = get_component_function(*RUN_PIPELINE)

    with pytest.raises(RuntimeError) as error:
        run_pipeline([str(labelled_csv)], {"output_dir": str(tmp_path / "out")})

    assert error.value.exit_code == 2
    assert not (tmp_path / "out").exists()


def test_failed_run_keeps_logs_and_metadata(labelled_csv, tmp_path, fake_gemini):
    run_pipeline = get_component_function(*RUN_PIPELINE)
    fake_gemini(_model_records(labelled_csv), mode="malformed")  # every Gemini answer is invalid

    with pytest.raises(RuntimeError) as error:
        run_pipeline([str(labelled_csv)], {"allow_cloud_processing": True, "output_dir": str(tmp_path / "out")})

    assert error.value.stage == "subcategory"
    assert error.value.exit_code == 5
    folder = Path(error.value.run_folder)
    metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed" and metadata["error"]["stage"] == "subcategory"
    assert "ERROR" in (folder / "logs.txt").read_text(encoding="utf-8")


def test_failed_api_check_stops_before_any_output(labelled_csv, tmp_path, monkeypatch):
    run_pipeline = get_component_function(*RUN_PIPELINE)
    from app_review_intelligence.components.subcategory import interface

    def refuse(*args, **kwargs):
        raise RuntimeError("Gemini API check failed: invalid key")

    monkeypatch.setattr(interface, "check_gemini_access", refuse)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)

    with pytest.raises(RuntimeError) as error:
        run_pipeline([str(labelled_csv)], {"allow_cloud_processing": True, "output_dir": str(tmp_path / "out")})

    assert error.value.exit_code == 5 and error.value.stage == "api_check"
    assert not (tmp_path / "out").exists()


def test_config_dir_is_used_by_the_pipeline(labelled_csv, tmp_path, fake_gemini):
    import shutil

    from conftest import CONFIG_DIR

    run_pipeline = get_component_function(*RUN_PIPELINE)
    custom = tmp_path / "custom_config"
    shutil.copytree(CONFIG_DIR, custom)
    fake_gemini(_model_records(labelled_csv))

    result = run_pipeline([str(labelled_csv)], {
        "allow_cloud_processing": True, "output_dir": str(tmp_path / "out"), "config_dir": str(custom),
    })

    assert result["run_metadata"]["status"] == "completed"
    (custom / "settings.yaml").unlink()
    with pytest.raises(RuntimeError) as error:
        run_pipeline([str(labelled_csv)], {"allow_cloud_processing": True, "config_dir": str(custom)})
    assert error.value.exit_code == 2
