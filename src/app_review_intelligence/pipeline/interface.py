"""Public interface for the pipeline. No logic lives here.

Example:
    from app_review_intelligence.pipeline.interface import run_pipeline

    result = run_pipeline(
        ["dataset/processed/combined_reviews_output.csv"],
        {"mode": "run", "allow_cloud_processing": True},
    )
    result["run_folder"]   # "outputs/run_20261115T101500Z_a1b2c3d4"
    result["report_path"]  # ".../report.md"
"""

from app_review_intelligence.pipeline.implementation import (
    api_key_env_var as _api_key_env_var,
    load_settings as _load_settings,
    run_pipeline as _run_pipeline,
)
from app_review_intelligence.pipeline.run_manager import (
    create_run_folder as _create_run_folder,
)


def run_pipeline(input_paths, config=None):
    """
    Runs the complete review analysis pipeline.

    Input:
        input_paths: path or list of paths to CSV/XLSX files.
        config: dict with optional keys mode ("run" | "evaluate"),
            allow_cloud_processing (must be True), config_dir, output_dir,
            id_column, text_column, date_column, sheet, seed, verbose.

    Returns:
        dict with run_folder, run_id, mode, report_path, classified_reviews,
        clusters, priority_results, run_metadata (+ evaluation_results in
        evaluate mode).

    Raises:
        ValueError / FileNotFoundError: no input or a missing input file.
        PipelineError (RuntimeError with .stage, .exit_code, .run_folder):
            any later failure. A failed run keeps its folder, logs.txt and
            run_metadata.json for diagnosis.
    """
    return _run_pipeline(input_paths, config)


def load_settings(config_dir=None):
    """Loads and validates config/settings.yaml (raises the pipeline error with exit code 2)."""
    return _load_settings(config_dir)


def api_key_env_var(settings):
    """Name of the environment variable that holds the Gemini API key."""
    return _api_key_env_var(settings)


def create_run_folder(folder_type, output_root="outputs"):
    """Creates a new unique run_/eval_ folder; never reuses an existing one."""
    return _create_run_folder(folder_type, output_root)
