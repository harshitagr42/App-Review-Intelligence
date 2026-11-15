"""Pipeline (Person 6): connects every component through its interface.py.

Order (writing_block.md section 4):
    input adapter + preprocessing -> major category -> sentiment ->
    subcategory (Gemini) -> priority -> evaluation (evaluate mode) ->
    reporting (run mode)

This module contains no component logic. It only:
- validates paths, consent and configuration,
- creates a new, unique output folder and writes logs.txt into it,
- calls the component interfaces in order and passes results along,
- records run metadata (always written, also for failed runs).
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import resource
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app_review_intelligence.pipeline.run_manager import create_run_folder

logger = logging.getLogger("app_review_intelligence.pipeline")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
SCHEMA_VERSION = "1.0"

# Fixed models (Docs/06). A configuration that names another model is rejected.
FIXED_MODEL_IDS = {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "llm": "gemini-3.6-flash",
}

REQUIRED_SETTINGS = (
    "input_batch_size",
    "embedding_batch_size",
    "sentiment_batch_size",
    "llm_batch_size",
    "category_min_similarity",
    "llm_cluster_merge_similarity",
    "random_seed",
    "max_process_memory_mb",
    "models",
    "priority",
)

GOLD_COLUMNS = ("Major Class", "Minor Class", "Sentiment")

# Exit codes (Docs/03 section 7).
EXIT_GENERAL = 1
EXIT_CONFIG = 2
EXIT_INPUT = 3
EXIT_MODEL = 4
EXIT_API = 5
EXIT_EVALUATION = 6
EXIT_MEMORY = 7

STAGE_EXIT_CODES = {
    "input": EXIT_INPUT,
    "major_category": EXIT_MODEL,
    "sentiment": EXIT_MODEL,
    "subcategory": EXIT_API,
    "priority": EXIT_GENERAL,
    "evaluation": EXIT_EVALUATION,
    "reporting": EXIT_GENERAL,
}

STAGE_SUGGESTIONS = {
    "input": "Check the file path, format (CSV/XLSX) and column-mapping options.",
    "major_category": "Check that the MiniLM model can be downloaded or is in the Hugging Face cache.",
    "sentiment": "Check that the CardiffNLP model can be downloaded or is in the Hugging Face cache.",
    "subcategory": "Check GEMINI_API_KEY, network access and Gemini quota, then retry.",
    "priority": "Check config/settings.yaml priority section.",
    "evaluation": "Check that the input has Major Class, Minor Class and Sentiment columns.",
    "reporting": "Check that the output directory is writable.",
}


class PipelineError(RuntimeError):
    """A pipeline failure tied to one stage, with an exit code and a suggested fix."""

    def __init__(self, stage, message, exit_code=EXIT_GENERAL, suggestion=None, run_folder=None):
        self.stage = stage
        self.exit_code = exit_code
        self.suggestion = suggestion
        self.run_folder = run_folder
        text = f"Stage '{stage}' failed: {message}"
        if suggestion:
            text += f" Suggested fix: {suggestion}"
        super().__init__(text)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_settings(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and validate config/settings.yaml. Raises PipelineError (exit 2)."""
    directory = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    path = directory / "settings.yaml"
    if not path.is_file():
        raise PipelineError(
            "configuration", f"settings file not found: {path}", EXIT_CONFIG,
            "Pass --config-dir pointing at a folder with settings.yaml.",
        )
    try:
        with path.open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle) or {}
    except yaml.YAMLError as error:
        raise PipelineError("configuration", f"{path.name} is not valid YAML: {error}", EXIT_CONFIG) from error

    if not isinstance(settings, dict):
        raise PipelineError("configuration", f"{path.name} must be a mapping", EXIT_CONFIG)
    missing = [key for key in REQUIRED_SETTINGS if key not in settings]
    if missing:
        raise PipelineError("configuration", f"{path.name} is missing keys: {missing}", EXIT_CONFIG)

    models = settings.get("models") or {}
    for role, expected in FIXED_MODEL_IDS.items():
        actual = (models.get(role) or {}).get("model_id")
        if actual != expected:
            raise PipelineError(
                "configuration",
                f"models.{role}.model_id must be {expected!r} (found {actual!r}); models are fixed",
                EXIT_CONFIG,
            )

    taxonomy = directory / "major_categories.yaml"
    if not taxonomy.is_file():
        raise PipelineError("configuration", f"taxonomy file not found: {taxonomy}", EXIT_CONFIG)

    try:
        with taxonomy.open(encoding="utf-8") as handle:
            categories = (yaml.safe_load(handle) or {}).get("categories") or []
        settings["_taxonomy_names"] = sorted(
            str(item["name"]) for item in categories if isinstance(item, dict) and item.get("name")
        )
    except (yaml.YAMLError, AttributeError) as error:
        raise PipelineError("configuration", f"{taxonomy.name} is not valid: {error}", EXIT_CONFIG) from error

    settings["_config_dir"] = str(directory)
    settings["_settings_sha256"] = _sha256_file(path)
    settings["_taxonomy_sha256"] = _sha256_file(taxonomy)
    return settings


def api_key_env_var(settings: dict[str, Any]) -> str:
    return ((settings.get("models") or {}).get("llm") or {}).get("api_key_env_var", "GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Helpers: paths, logging, memory, model revisions
# ---------------------------------------------------------------------------

def _normalize_paths(input_paths) -> list[Path]:
    if input_paths is None:
        raise ValueError("At least one input file must be provided.")
    if isinstance(input_paths, (str, Path)):
        paths = [Path(input_paths)]
    else:
        paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one input file must be provided.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    return paths


class _RedactFilter(logging.Filter):
    """Removes the API key value from every log record."""

    def __init__(self, env_var):
        super().__init__()
        self.env_var = env_var

    def filter(self, record):
        secret = os.environ.get(self.env_var)
        if secret:
            message = record.getMessage().replace(secret, "[REDACTED]")
            record.msg, record.args = message, None
        return True


def _attach_log_file(folder: Path, verbose: bool, env_var: str) -> tuple[logging.Handler, int]:
    package_logger = logging.getLogger("app_review_intelligence")
    handler = logging.FileHandler(folder / "logs.txt", mode="x", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.addFilter(_RedactFilter(env_var))
    previous_level = package_logger.level
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return handler, previous_level


def _detach_log_file(handler: logging.Handler, previous_level: int) -> None:
    package_logger = logging.getLogger("app_review_intelligence")
    package_logger.removeHandler(handler)
    package_logger.setLevel(previous_level)
    handler.close()


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def _model_revision(model_id: str) -> str | None:
    """Commit hash of the model in the project's models/ folder (or the HF cache as fallback)."""
    from app_review_intelligence.model_store.interface import model_revision

    revision = model_revision(model_id)
    if revision:
        return revision
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None
    ref = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_paths, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Run the complete review analysis pipeline.

    config keys (all optional):
        mode: "run" (default) writes the run report; "evaluate" scores the
            predictions against gold labels and writes the evaluation report.
        allow_cloud_processing: must be True; review text is sent to Gemini.
        config_dir, output_dir, id_column, text_column, date_column, sheet,
        seed, verbose.

    Returns run information and output file locations.
    Raises ValueError / FileNotFoundError for bad input paths, and
    PipelineError (with .stage, .exit_code, .run_folder) for later failures.
    """
    options = dict(config or {})
    paths = _normalize_paths(input_paths)

    mode = options.get("mode", "run")
    if mode not in ("run", "evaluate"):
        raise ValueError("mode must be 'run' or 'evaluate'")

    if not options.get("allow_cloud_processing"):
        raise PipelineError(
            "consent", "cloud processing was not allowed; review text would be sent to Gemini",
            EXIT_CONFIG, "Re-run with --allow-cloud-processing.",
        )

    settings = load_settings(options.get("config_dir"))
    env_var = api_key_env_var(settings)
    if not os.environ.get(env_var):
        raise PipelineError(
            "configuration", f"{env_var} is not set", EXIT_CONFIG,
            f"Set {env_var} in your environment or in the .env file.",
        )

    seed = int(options.get("seed", settings.get("random_seed", 42)))
    output_root = Path(
        options.get("output_dir")
        or options.get("output_root")
        or (settings.get("output") or {}).get("directory", "outputs")
    )
    column_mapping = {
        "id": options.get("id_column") or "ID",
        "text": options.get("text_column") or "Review",
        "date": options.get("date_column") or "Date",
    }
    sheet = options.get("sheet")

    # Pre-flight Gemini check: key, network and model ID. Sends no review text.
    if options.get("check_api_access", True):
        from app_review_intelligence.components.subcategory.interface import check_gemini_access

        llm_settings = settings["models"]["llm"]
        try:
            check_gemini_access(
                llm_settings["model_id"], env_var, llm_settings.get("timeout_seconds", 60),
            )
        except Exception as error:
            raise PipelineError(
                "api_check", str(error), EXIT_API,
                f"Check that {env_var} is valid, the network is reachable and the Gemini quota is not exhausted.",
            ) from error

    from app_review_intelligence.adapters.input.interface import load_and_prepare

    # 1. Input adapter + preprocessing. Runs before the output folder exists, so
    #    an invalid file never leaves an empty run folder behind.
    started = time.perf_counter()
    started_at = _utc_now()
    try:
        model_records, evaluation_labels, dataset_metadata = load_and_prepare(
            [str(path) for path in paths],
            id_column=column_mapping["id"],
            text_column=column_mapping["text"],
            date_column=column_mapping["date"],
            sheet_name=sheet,
        )
    except (ValueError, FileNotFoundError) as error:
        raise PipelineError("input", str(error), EXIT_INPUT, STAGE_SUGGESTIONS["input"]) from error
    input_seconds = time.perf_counter() - started

    if mode == "evaluate":
        found = set(dataset_metadata.get("gold_label_columns_found") or [])
        missing = [column for column in GOLD_COLUMNS if column not in found]
        if missing:
            raise PipelineError(
                "evaluation", f"input is missing gold-label column(s): {missing}",
                EXIT_EVALUATION, "Evaluate with a labelled file such as combined_reviews_output.csv.",
            )
    else:
        # `run` ignores gold labels entirely.
        evaluation_labels = []

    folder = create_run_folder("eval" if mode == "evaluate" else "run", output_root)
    handler, previous_level = _attach_log_file(folder, bool(options.get("verbose")), env_var)
    _seed_everything(seed)

    stage_durations = {"input": round(input_seconds, 3)}
    warnings = list(dataset_metadata.get("warnings") or [])
    model_ids = {
        "embedding": FIXED_MODEL_IDS["embedding"],
        "sentiment": FIXED_MODEL_IDS["sentiment"],
        "llm": settings["models"]["llm"]["model_id"],
    }
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": folder.name,
        "mode": mode,
        "status": "running",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "duration_seconds": None,
        "input_files": [
            {"file_name": path.name, "sha256": _sha256_file(path)} for path in paths
        ],
        "worksheet_names": [
            item.get("sheet_name") for item in dataset_metadata.get("input_files", [])
        ] if any(item.get("sheet_name") for item in dataset_metadata.get("input_files", [])) else None,
        "column_mapping": dataset_metadata.get("column_mapping", column_mapping),
        "rows_read": dataset_metadata.get("rows_loaded"),
        "rows_accepted": dataset_metadata.get("rows_accepted"),
        "rows_skipped_empty": dataset_metadata.get("rows_skipped_empty"),
        "rows_skipped_invalid_date": dataset_metadata.get("rows_skipped_invalid_date"),
        "duplicate_review_count": dataset_metadata.get("duplicate_review_count"),
        "embedding_model_id": model_ids["embedding"],
        "embedding_model_revision": _model_revision(model_ids["embedding"]),
        "sentiment_model_id": model_ids["sentiment"],
        "sentiment_model_revision": _model_revision(model_ids["sentiment"]),
        "llm_model_id": model_ids["llm"],
        "category_taxonomy_sha256": settings["_taxonomy_sha256"],
        "settings_sha256": settings["_settings_sha256"],
        "random_seed": seed,
        "stage_durations_seconds": stage_durations,
        "peak_process_rss_mb": None,
        "api_request_count": 0,
        "api_retry_count": 0,
        "unassigned_review_count": None,
        "priority_available": False,
        "warnings": warnings,
        "error": None,
    }
    memory_limit = float(settings.get("max_process_memory_mb", 5120))
    current_stage = {"name": "setup"}

    @contextmanager
    def stage(name):
        current_stage["name"] = name
        logger.info("Stage %s started", name)
        begin = time.perf_counter()
        yield
        stage_durations[name] = round(time.perf_counter() - begin, 3)
        peak = _peak_rss_mb()
        metadata["peak_process_rss_mb"] = peak
        logger.info("Stage %s finished in %.2fs (peak RSS %.1f MB)", name, stage_durations[name], peak)
        if peak > memory_limit:
            raise PipelineError(
                "memory", f"peak RSS {peak:.0f} MB exceeded the {memory_limit:.0f} MB limit after stage '{name}'",
                EXIT_MEMORY, "Use a smaller input or close other applications.",
            )

    from app_review_intelligence.reporting.interface import write_run_metadata

    result: dict[str, Any] = {"run_folder": str(folder), "run_id": folder.name, "mode": mode}

    try:
        logger.info(
            "Run %s (%s): %d file(s), %d rows read, %d accepted",
            folder.name, mode, len(paths), metadata["rows_read"] or 0, metadata["rows_accepted"] or 0,
        )
        for message in warnings:
            logger.warning("Input: %s", message)

        from app_review_intelligence.components.major_category.interface import (
            classify,
            unload_model as unload_embedding_model,
        )
        from app_review_intelligence.components.sentiment.interface import (
            analyze,
            unload_model as unload_sentiment_model,
        )
        from app_review_intelligence.components.subcategory.interface import (
            discover_subcategories_with_report,
        )

        # 2. Major category (MiniLM). Released before the sentiment model loads.
        with stage("major_category"):
            try:
                major_predictions = classify(
                    model_records,
                    config_path=str(Path(settings["_config_dir"]) / "major_categories.yaml"),
                )
            finally:
                unload_embedding_model()

        # 3. Sentiment (CardiffNLP), independent of the category steps.
        with stage("sentiment"):
            try:
                sentiment_results = analyze(model_records)
            finally:
                unload_sentiment_model()

        # 4. Subcategory discovery. Gemini receives only ID, text and predicted category.
        major_by_id = {item["review_id"]: item for item in major_predictions}
        gemini_records = [
            {
                "review_id": record["review_id"],
                "review_text": record["review_text"],
                "predicted_major_category": major_by_id[record["review_id"]]["predicted_major_category"],
            }
            for record in model_records
        ]
        llm = settings["models"]["llm"]
        subcategory_overrides = {
            "model_id": llm["model_id"],
            "api_key_env_var": env_var,
            "batch_size": llm.get("batch_size", settings["llm_batch_size"]),
            "max_retries": llm.get("max_retries", 3),
            "merge_similarity": settings["llm_cluster_merge_similarity"],
            "embedding_model_id": FIXED_MODEL_IDS["embedding"],
            "random_seed": seed,
            "config_dir": settings["_config_dir"],
        }
        with stage("subcategory"):
            subcategory_result, subcategory_report = discover_subcategories_with_report(
                gemini_records, config=subcategory_overrides,
            )
        metadata["api_request_count"] = subcategory_report.get("api_calls", 0)
        metadata["api_retry_count"] = subcategory_report.get("retries", 0)
        unassigned_ids = list(subcategory_report.get("unassigned_review_ids") or [])
        metadata["unassigned_review_count"] = len(unassigned_ids)

        # 5. Priority (Person 5). Skipped with a warning while it is not implemented.
        priority_results: list[dict[str, Any]] = []
        with stage("priority"):
            try:
                from app_review_intelligence.components.priority.interface import calculate_priorities
            except ModuleNotFoundError as error:
                if not (error.name or "").startswith("app_review_intelligence.components.priority"):
                    raise
                message = "priority component is not implemented yet; priority scores are omitted"
                logger.warning(message)
                warnings.append(message)
            else:
                priority_results = calculate_priorities(subcategory_result, sentiment_results, model_records)
                metadata["priority_available"] = True

        classified_reviews = _classified_rows(
            model_records, dataset_metadata, major_by_id, sentiment_results,
            subcategory_result, priority_results,
        )

        result.update({
            "classified_reviews": classified_reviews,
            "clusters": subcategory_result,
            "priority_results": priority_results,
        })

        if mode == "evaluate":
            # 6. Evaluation. Gold labels are used only here.
            from app_review_intelligence.evaluation.interface import evaluate_predictions

            with stage("evaluation"):
                evaluation_results = evaluate_predictions(
                    major_predictions=major_predictions,
                    sentiment_results=sentiment_results,
                    subcategory_assignments=subcategory_result.get("assignments", []),
                    evaluation_labels=evaluation_labels,
                    output_dir=folder,
                    run_metadata=_metadata_snapshot(metadata, started),
                    priority_results=priority_results,
                    excluded_records=_excluded_records(dataset_metadata),
                    seed=seed,
                    model_ids=model_ids,
                    runtime_seconds=time.perf_counter() - started,
                    memory_mb=_peak_rss_mb(),
                    known_major_categories=set(settings["_taxonomy_names"]),
                    known_sentiments={"Positive", "Negative", "Neutral"},
                )
            result["evaluation_results"] = evaluation_results
            result["report_path"] = str(folder / "evaluation_report.md")
            summary = evaluation_results.get("evaluation_summary", {})
            if summary.get("status") != "available":
                warnings.append(f"evaluation status is {summary.get('status')!r}")
        else:
            # 7. Reporting: report.md, classified_reviews.csv, clusters.json.
            from app_review_intelligence.reporting.interface import generate_report

            with stage("reporting"):
                report_path = generate_report(
                    {
                        "priorities": priority_results,
                        "unassigned_review_ids": unassigned_ids,
                        "warnings": warnings,
                    },
                    folder,
                    classified_reviews=classified_reviews,
                    clusters=subcategory_result,
                    run_metadata=_metadata_snapshot(metadata, started),
                )
            result["report_path"] = report_path

        metadata["status"] = "completed"
        logger.info("Run %s completed", folder.name)

    except Exception as error:
        failed_stage = current_stage["name"]
        if isinstance(error, PipelineError):
            failure = error
        elif isinstance(error, MemoryError):
            failure = PipelineError(failed_stage, "out of memory", EXIT_MEMORY, "Use a smaller input.")
        else:
            failure = PipelineError(
                failed_stage, f"{type(error).__name__}: {error}",
                STAGE_EXIT_CODES.get(failed_stage, EXIT_GENERAL),
                STAGE_SUGGESTIONS.get(failed_stage),
            )
        failure.run_folder = str(folder)
        metadata["status"] = "failed"
        metadata["error"] = {
            "stage": failure.stage,
            "message": str(failure),
            "exit_code": failure.exit_code,
        }
        logger.error("Run failed: %s", failure)
        if options.get("verbose"):
            logger.error("Traceback:\n%s", "".join(traceback.format_exception(error)))
        raise failure from error

    finally:
        snapshot = _metadata_snapshot(metadata, started)
        snapshot["finished_at_utc"] = _utc_now()
        try:
            write_run_metadata(folder, snapshot)
        except Exception as write_error:  # never hide the original failure
            logger.error("Could not write run_metadata.json: %s", write_error)
        _detach_log_file(handler, previous_level)

    result["run_metadata"] = snapshot
    return result


def _metadata_snapshot(metadata: dict[str, Any], started: float) -> dict[str, Any]:
    snapshot = dict(metadata)
    snapshot["duration_seconds"] = round(time.perf_counter() - started, 3)
    snapshot["stage_durations_seconds"] = dict(metadata["stage_durations_seconds"])
    snapshot["warnings"] = list(metadata["warnings"])
    snapshot["peak_process_rss_mb"] = _peak_rss_mb()
    return snapshot


def _excluded_records(dataset_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    report = dataset_metadata.get("preprocessing") or {}
    excluded = [
        {"review_id": review_id, "reason": "empty review"}
        for review_id in report.get("skipped_empty_text", [])
    ]
    excluded += [
        {"review_id": item.get("review_id"), "reason": "empty or invalid date"}
        for item in report.get("skipped_invalid_date", [])
    ]
    excluded += [
        {
            "review_id": item.get("review_id"),
            "reason": "duplicate text and date",
            "duplicate_of_review_id": item.get("duplicate_of_review_id"),
        }
        for item in report.get("duplicates_removed", [])
    ]
    return excluded


def _classified_rows(
    model_records, dataset_metadata, major_by_id, sentiment_results,
    subcategory_result, priority_results,
) -> list[dict[str, Any]]:
    """One row per accepted review: source columns + every prediction (no gold labels)."""
    sources = dataset_metadata.get("record_sources") or {}
    sentiment_by_id = {item["review_id"]: item for item in sentiment_results}
    assignment_by_id = {
        item["review_id"]: item["subcategory_id"]
        for item in subcategory_result.get("assignments", [])
    }
    subcategory_by_id = {
        item["subcategory_id"]: item for item in subcategory_result.get("subcategories", [])
    }
    priority_by_id = {item.get("subcategory_id"): item for item in priority_results}

    rows = []
    for record in model_records:
        review_id = record["review_id"]
        source = sources.get(review_id, {})
        major = major_by_id.get(review_id, {})
        sentiment = sentiment_by_id.get(review_id, {})
        subcategory_id = assignment_by_id.get(review_id, "UNASSIGNED")
        priority = priority_by_id.get(subcategory_id, {})
        rows.append({
            "review_id": review_id,
            "source_file": source.get("source_file"),
            "source_review_id": source.get("source_review_id"),
            "source_row_number": source.get("source_row"),
            "review_text": record["review_text"],
            "review_date": record.get("review_date"),
            "predicted_major_category": major.get("predicted_major_category"),
            "major_category_confidence": major.get("confidence"),
            "subcategory_id": subcategory_id,
            "subcategory_name": subcategory_by_id.get(subcategory_id, {}).get("subcategory_name"),
            "sentiment": sentiment.get("sentiment"),
            "sentiment_confidence": sentiment.get("confidence"),
            "priority_score": priority.get("priority_score"),
            "priority_level": priority.get("priority_level"),
        })
    return rows
