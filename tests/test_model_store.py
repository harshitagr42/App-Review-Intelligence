"""Tests for model_store (local models in models/) and the sentiment text normalisation."""

import pytest

from conftest import get_component_function


STORE = "app_review_intelligence.model_store.interface"


def test_only_the_two_local_models_are_managed():
    local_model_ids = get_component_function(STORE, "local_model_ids")

    assert local_model_ids() == [
        "sentence-transformers/all-MiniLM-L6-v2",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ]


def test_unknown_model_id_is_returned_unchanged():
    resolve_model_path = get_component_function(STORE, "resolve_model_path")

    assert resolve_model_path("gemini-3.6-flash") == "gemini-3.6-flash"


def test_models_dir_can_be_overridden(monkeypatch, tmp_path):
    models_dir = get_component_function(STORE, "models_dir")
    is_downloaded = get_component_function(STORE, "is_downloaded")
    model_revision = get_component_function(STORE, "model_revision")
    monkeypatch.setenv("APP_REVIEW_INTELLIGENCE_MODELS_DIR", str(tmp_path))

    assert models_dir() == tmp_path.resolve()
    assert not is_downloaded("sentence-transformers/all-MiniLM-L6-v2")
    assert model_revision("sentence-transformers/all-MiniLM-L6-v2") is None


def test_models_folder_is_git_ignored():
    import subprocess

    from conftest import REPO_ROOT

    result = subprocess.run(
        ["git", "check-ignore", "-q", "models/anything"], cwd=REPO_ROOT, capture_output=True,
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("@support_team fix this http://x.co/abc now!!", "@user fix this http now!!"),
        ("I don't like it 😡", "I don't like it 😡"),
        ("@ alone stays", "@ alone stays"),
    ],
)
def test_sentiment_normalisation_follows_model_card(text, expected):
    from app_review_intelligence.components.sentiment.implementation import normalize_text

    assert normalize_text(text) == expected
