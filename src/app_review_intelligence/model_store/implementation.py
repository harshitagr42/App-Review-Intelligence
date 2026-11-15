"""Local model store: keeps the fixed Hugging Face models in the project's `models/` folder.

Every local model is downloaded once into `models/<org>--<name>/` (git-ignored)
and always loaded from that folder afterwards, so runs work offline and never
depend on the global Hugging Face cache. The commit hash that was downloaded is
saved in `models/<org>--<name>/REVISION` for run metadata.

The folder can be moved with the APP_REVIEW_INTELLIGENCE_MODELS_DIR variable.
Only the PyTorch weights and tokenizer/config files are downloaded (no ONNX,
OpenVINO, TensorFlow, Flax or Rust weights).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR_ENV_VAR = "APP_REVIEW_INTELLIGENCE_MODELS_DIR"

# The only local models the project uses (Docs/06). Gemini runs in the cloud.
LOCAL_MODEL_IDS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)

IGNORE_PATTERNS = [
    "*.h5", "*.ot", "*.msgpack", "*.onnx", "*.tflite",
    "onnx/*", "openvino/*", "tf_*", "flax_*", "rust_*",
]

_REVISION_FILE = "REVISION"
_COMPLETE_MARKER = ".download_complete"
_lock = threading.Lock()


def models_dir() -> Path:
    override = os.environ.get(MODELS_DIR_ENV_VAR)
    return Path(override).expanduser().resolve() if override else REPO_ROOT / "models"


def local_model_dir(model_id: str) -> Path:
    return models_dir() / model_id.replace("/", "--")


def is_downloaded(model_id: str) -> bool:
    return (local_model_dir(model_id) / _COMPLETE_MARKER).is_file()


def model_revision(model_id: str) -> str | None:
    try:
        return (local_model_dir(model_id) / _REVISION_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def download_model(model_id: str, force: bool = False) -> Path:
    """Download model_id into models/ (no-op if already complete). Returns the folder."""
    if model_id not in LOCAL_MODEL_IDS:
        raise ValueError(f"{model_id!r} is not one of the project's local models {LOCAL_MODEL_IDS}")
    target = local_model_dir(model_id)
    with _lock:
        if is_downloaded(model_id) and not force:
            return target
        try:
            from huggingface_hub import HfApi, snapshot_download

            info = HfApi().model_info(model_id, files_metadata=False)
            revision = info.sha
            ignore = list(IGNORE_PATTERNS)
            if any(item.rfilename.endswith(".safetensors") for item in info.siblings or []):
                ignore.append("*.bin")  # safetensors already holds the same weights
            logger.info("Downloading %s (revision %s) into %s", model_id, revision, target)
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(target),
                ignore_patterns=ignore,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not download model {model_id!r}: {error}. "
                "Check the network connection, then run `python inference.py download-models`."
            ) from error
        (target / _REVISION_FILE).write_text(f"{revision or ''}\n", encoding="utf-8")
        (target / _COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
        return target


def resolve_model_path(model_id: str) -> str:
    """
    Local folder to load model_id from, downloading it first if needed.

    Model IDs that are not project models are returned unchanged.
    """
    if model_id not in LOCAL_MODEL_IDS:
        return model_id
    return str(download_model(model_id))


def download_all(force: bool = False) -> dict[str, str]:
    return {model_id: str(download_model(model_id, force=force)) for model_id in LOCAL_MODEL_IDS}
