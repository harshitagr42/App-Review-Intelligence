"""Public interface for the local model store. No logic lives here.

Example:
    from app_review_intelligence.model_store.interface import resolve_model_path

    path = resolve_model_path("sentence-transformers/all-MiniLM-L6-v2")
    # ".../models/sentence-transformers--all-MiniLM-L6-v2" (downloaded on first use)
"""

from app_review_intelligence.model_store.implementation import (
    LOCAL_MODEL_IDS,
    download_all as _download_all,
    is_downloaded as _is_downloaded,
    model_revision as _model_revision,
    models_dir as _models_dir,
    resolve_model_path as _resolve_model_path,
)


def resolve_model_path(model_id):
    """Returns the local folder to load model_id from; downloads it into models/ if missing."""
    return _resolve_model_path(model_id)


def download_all_models(force=False):
    """Downloads every local project model into models/. Returns {model_id: folder}."""
    return _download_all(force)


def is_downloaded(model_id):
    """True when model_id is fully present in models/."""
    return _is_downloaded(model_id)


def model_revision(model_id):
    """Commit hash of the downloaded model, or None if it is not downloaded."""
    return _model_revision(model_id)


def models_dir():
    """Folder that holds the local models (default: <project>/models)."""
    return _models_dir()


def local_model_ids():
    """The project's local model IDs (Gemini is not included; it runs in the cloud)."""
    return list(LOCAL_MODEL_IDS)
