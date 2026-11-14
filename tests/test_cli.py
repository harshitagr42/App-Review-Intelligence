"""
Tests for the CLI entry point `python inference.py` (Person 6).
Spec: Docs/09_CLI_AND_REPORT_SPECIFICATION.md and Docs/03_APP_FLOW.md.

Skipped until inference.py exists. These tests never send data to Gemini:
they only exercise help, argument checks and input validation.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from conftest import REPO_ROOT


INFERENCE = REPO_ROOT / "inference.py"
EXIT_INVALID_INPUT = 3  # Docs/03 section 7

pytestmark = pytest.mark.skipif(not INFERENCE.exists(), reason="inference.py is not implemented yet")


def _run(*args, env=None):
    environment = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, str(INFERENCE), *args],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _outputs_snapshot():
    outputs = REPO_ROOT / "outputs"
    return set(outputs.iterdir()) if outputs.exists() else set()


@pytest.fixture
def small_csv(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame([
        {"ID": "1", "Review": "The app crashes on login.", "Date": "2026-07-21"},
        {"ID": "2", "Review": "Great app, easy to use.", "Date": "2026-07-22"},
    ]).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("command", [[], ["validate"], ["run"], ["evaluate"]])
def test_help_works_for_root_and_every_command(command):
    result = _run(*command, "--help")

    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout


def test_validate_succeeds_without_creating_outputs(small_csv):
    before = _outputs_snapshot()

    result = _run("validate", "--input", str(small_csv))

    assert result.returncode == 0, result.stderr
    assert _outputs_snapshot() == before


def test_validate_rejects_unsupported_file(tmp_path):
    path = tmp_path / "reviews.txt"
    path.write_text("ID,Review\n1,hello\n")

    result = _run("validate", "--input", str(path))

    assert result.returncode == EXIT_INVALID_INPUT


def test_validate_rejects_missing_text_column(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame([{"ID": "1", "Text": "wrong column"}]).to_csv(path, index=False)

    result = _run("validate", "--input", str(path))

    assert result.returncode == EXIT_INVALID_INPUT
    assert "Review" in result.stdout + result.stderr


def test_errors_do_not_print_tracebacks(tmp_path):
    path = tmp_path / "reviews.txt"
    path.write_text("ID,Review\n1,hello\n")

    result = _run("validate", "--input", str(path))

    assert "Traceback" not in result.stdout + result.stderr


def test_run_refuses_without_cloud_consent(small_csv):
    before = _outputs_snapshot()

    result = _run("run", "--input", str(small_csv), env={"GEMINI_API_KEY": "test-key-NOT-A-REAL-KEY"})

    assert result.returncode != 0
    assert "allow-cloud-processing" in result.stdout + result.stderr
    assert _outputs_snapshot() == before


def test_run_refuses_without_api_key(small_csv):
    result = _run("run", "--input", str(small_csv), "--allow-cloud-processing")

    assert result.returncode != 0
    assert "GEMINI_API_KEY" in result.stdout + result.stderr


def test_api_key_is_not_accepted_as_an_argument(small_csv):
    result = _run("run", "--input", str(small_csv), "--api-key", "abc", "--allow-cloud-processing")

    assert result.returncode != 0
