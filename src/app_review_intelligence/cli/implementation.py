"""CLI (Person 6): `python inference.py validate | run | evaluate`.

Spec: Docs/09_CLI_AND_REPORT_SPECIFICATION.md section 2, Docs/03_APP_FLOW.md.
The CLI only parses arguments, checks consent and the API key, and calls the
pipeline / input adapter / evaluation interfaces. It never prints tracebacks
or the API key.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

REPO_ROOT = Path(__file__).resolve().parents[3]
DOTENV_DISABLE_VAR = "APP_REVIEW_INTELLIGENCE_NO_DOTENV"

EXIT_OK = 0
EXIT_GENERAL = 1
EXIT_CONFIG = 2
EXIT_INPUT = 3
EXIT_MODEL = 4
EXIT_API = 5
EXIT_EVALUATION = 6
EXIT_MEMORY = 7
INVALID_INPUT_EXIT_CODE = EXIT_INPUT

EXIT_CODE_HELP = (
    "Exit codes: 0 success; 1 general failure; 2 invalid arguments or configuration; "
    "3 invalid or unsupported input file; 4 model download or local inference failure; "
    "5 Gemini API/network/quota failure; 6 evaluation dataset or metric failure; "
    "7 memory safety limit reached."
)

PRIVACY_WARNING = (
    "Privacy notice: review IDs, review text and predicted major categories will be sent to "
    "the Google Gemini API. On the free tier, Google may use submitted content to improve its "
    "products. Do not process confidential reviews without permission."
)

app = typer.Typer(
    help="App Review Intelligence: classify app reviews and prioritise the issues they raise.",
    epilog=EXIT_CODE_HELP,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _error(message: str, code: int) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=code)


def load_dotenv_file() -> list[str]:
    """
    Load KEY=VALUE pairs from .env (current directory, then the project root).

    Existing environment variables win; empty values are ignored. Values are
    never printed. Returns the names of variables that were set.
    """
    if os.environ.get(DOTENV_DISABLE_VAR) == "1":
        return []
    loaded = []
    for candidate in dict.fromkeys([Path.cwd() / ".env", REPO_ROOT / ".env"]):
        if not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if name and value and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
        break
    return loaded


def _configure_console_logging(quiet: bool, verbose: bool) -> None:
    package_logger = logging.getLogger("app_review_intelligence")
    for handler in list(package_logger.handlers):
        if getattr(handler, "_ari_console", False):
            package_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler._ari_console = True
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.setLevel(logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    package_logger.propagate = False


def _adapter_kwargs(id_column, text_column, date_column, sheet) -> dict:
    """Only pass column options that differ from the adapter defaults."""
    kwargs = {}
    if id_column != "ID":
        kwargs["id_column"] = id_column
    if text_column != "Review":
        kwargs["text_column"] = text_column
    if date_column != "Date":
        kwargs["date_column"] = date_column
    if sheet:
        kwargs["sheet_name"] = sheet
    return kwargs


def _check_input_files(paths: list[Path], text_column: str = "Review", sheet: Optional[str] = None) -> None:
    """Existence, extension and text-column checks. Exits with code 3 on failure."""
    if not paths:
        raise _error("at least one --input file is required.", EXIT_INPUT)
    for path in paths:
        if not path.is_file():
            raise _error(f"input file does not exist: {path}", EXIT_INPUT)
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".xlsx"}:
            raise _error(f"unsupported file format '{suffix or path.name}'. Use CSV or XLSX.", EXIT_INPUT)
        try:
            if suffix == ".csv":
                frame = pd.read_csv(path, nrows=5, dtype=str)
            else:
                frame = pd.read_excel(path, nrows=5, dtype=str, sheet_name=sheet or 0)
        except Exception as exc:
            raise _error(f"could not read {path.name}: {exc}", EXIT_INPUT)
        if text_column not in frame.columns:
            raise _error(
                f"{path.name} must contain a '{text_column}' column "
                "(use --text-column to map another column name).",
                EXIT_INPUT,
            )


# Kept for compatibility with existing callers/tests.
def validate_input_file(input_path: Path) -> None:
    _check_input_files([input_path])


def _shared_options():
    return {
        "input": typer.Option(..., "--input", help="CSV or XLSX review file. Repeat to combine several files."),
        "text_column": typer.Option("Review", "--text-column", help="Review-text column name."),
        "id_column": typer.Option("ID", "--id-column", help="Review-ID column name."),
        "date_column": typer.Option("Date", "--date-column", help="Review-date column name."),
        "sheet": typer.Option(None, "--sheet", help="XLSX worksheet (default: first sheet)."),
    }


_OPT = _shared_options()


def _run_pipeline_command(
    mode: str,
    inputs: list[Path],
    allow_cloud_processing: bool,
    text_column: str,
    id_column: str,
    date_column: str,
    sheet: Optional[str],
    config_dir: Path,
    output_dir: Optional[Path],
    quiet: bool,
    verbose: bool,
    seed: Optional[int] = None,
) -> dict:
    from app_review_intelligence.pipeline.interface import api_key_env_var, load_settings, run_pipeline

    if quiet and verbose:
        raise _error("use either --quiet or --verbose, not both.", EXIT_CONFIG)

    # Consent first: nothing is read, created or sent without it.
    if not allow_cloud_processing:
        raise _error(
            "this command sends review text to Gemini. Re-run with --allow-cloud-processing to consent.",
            EXIT_CONFIG,
        )

    load_dotenv_file()

    try:
        settings = load_settings(config_dir)
    except Exception as exc:
        raise _error(str(exc), getattr(exc, "exit_code", EXIT_CONFIG))

    env_var = api_key_env_var(settings)
    if not os.environ.get(env_var):
        raise _error(
            f"{env_var} is not set. Add it to the .env file (see .env.example) or export it: "
            f"macOS/Linux `export {env_var}=...`, Windows PowerShell `$env:{env_var}=\"...\"`.",
            EXIT_CONFIG,
        )

    _check_input_files(inputs, text_column, sheet)
    _configure_console_logging(quiet, verbose)
    if not quiet:
        typer.echo(PRIVACY_WARNING, err=True)

    config = {
        "mode": mode,
        "allow_cloud_processing": True,
        "config_dir": str(config_dir),
        "id_column": id_column,
        "text_column": text_column,
        "date_column": date_column,
        "sheet": sheet,
        "verbose": verbose,
    }
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if seed is not None:
        config["seed"] = seed

    try:
        return run_pipeline([str(path) for path in inputs], config)
    except FileNotFoundError as exc:
        raise _error(str(exc), EXIT_INPUT)
    except ValueError as exc:
        raise _error(f"invalid input: {exc}", EXIT_INPUT)
    except KeyboardInterrupt:
        raise _error("interrupted by user.", EXIT_GENERAL)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        run_folder = getattr(exc, "run_folder", None)
        if run_folder:
            typer.echo(f"Diagnostics kept in: {run_folder} (logs.txt, run_metadata.json)", err=True)
        raise typer.Exit(code=getattr(exc, "exit_code", EXIT_GENERAL))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command(epilog=EXIT_CODE_HELP)
def validate(
    input: list[Path] = _OPT["input"],
    text_column: str = _OPT["text_column"],
    id_column: str = _OPT["id_column"],
    date_column: str = _OPT["date_column"],
    sheet: Optional[str] = _OPT["sheet"],
) -> None:
    """Check input files without loading models, calling Gemini or creating outputs."""
    _check_input_files(input, text_column, sheet)

    from app_review_intelligence.adapters.input.interface import load_and_prepare

    # Warnings are printed once in the summary below, not also by the logger.
    logging.getLogger("app_review_intelligence").addHandler(logging.NullHandler())
    logging.getLogger("app_review_intelligence").propagate = False

    try:
        _, labels, meta = load_and_prepare(
            [str(path) for path in input], **_adapter_kwargs(id_column, text_column, date_column, sheet)
        )
    except (ValueError, FileNotFoundError) as exc:
        raise _error(str(exc), EXIT_INPUT)

    gold = meta.get("gold_label_columns_found") or []
    typer.echo("Input is valid.")
    for item in meta.get("input_files", []):
        sheet_note = f", sheet '{item['sheet_name']}'" if item.get("sheet_name") else ""
        typer.echo(f"  File: {item.get('file_name')} ({item.get('row_count')} rows{sheet_note})")
    typer.echo(f"  Rows read: {meta.get('rows_loaded')}")
    typer.echo(f"  Valid (accepted): {meta.get('rows_accepted')}")
    typer.echo(f"  Empty review: {meta.get('rows_skipped_empty')}")
    typer.echo(f"  Empty or invalid date: {meta.get('rows_skipped_invalid_date')}")
    typer.echo(f"  Duplicate text+date: {meta.get('duplicate_review_count')}")
    typer.echo(f"  Review IDs generated or de-duplicated: {len(meta.get('id_corrections') or [])}")
    typer.echo(f"  Gold-label columns: {', '.join(gold) if gold else 'none'}")
    for message in meta.get("warnings") or []:
        typer.echo(f"  Warning: {message}")


@app.command(epilog=EXIT_CODE_HELP)
def run(
    input: list[Path] = _OPT["input"],
    text_column: str = _OPT["text_column"],
    id_column: str = _OPT["id_column"],
    date_column: str = _OPT["date_column"],
    sheet: Optional[str] = _OPT["sheet"],
    allow_cloud_processing: bool = typer.Option(
        False, "--allow-cloud-processing", help="Required. Consent to sending review text to Gemini.",
    ),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", help="Folder with settings.yaml."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output root (default: outputs/)."),
    quiet: bool = typer.Option(False, "--quiet", help="Only show warnings and errors."),
    verbose: bool = typer.Option(False, "--verbose", help="Debug output; tracebacks go to logs.txt."),
) -> None:
    """Run the full pipeline and write a new outputs/run_<UTC>_<id>/ folder."""
    result = _run_pipeline_command(
        "run", input, allow_cloud_processing, text_column, id_column, date_column, sheet,
        config_dir, output_dir, quiet, verbose,
    )
    typer.echo("Pipeline completed successfully.")
    typer.echo(f"Run folder: {result['run_folder']}")
    typer.echo(f"Report: {result['report_path']}")


@app.command(epilog=EXIT_CODE_HELP)
def evaluate(
    input: list[Path] = _OPT["input"],
    text_column: str = _OPT["text_column"],
    id_column: str = _OPT["id_column"],
    date_column: str = _OPT["date_column"],
    sheet: Optional[str] = _OPT["sheet"],
    allow_cloud_processing: bool = typer.Option(
        False, "--allow-cloud-processing", help="Required for a full evaluation run (sends text to Gemini).",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", help="Folder with settings.yaml."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output root (default: outputs/)."),
    quiet: bool = typer.Option(False, "--quiet", help="Only show warnings and errors."),
    verbose: bool = typer.Option(False, "--verbose", help="Debug output; tracebacks go to logs.txt."),
    run_dir: Optional[Path] = typer.Option(
        None, "--run-dir",
        help="Optional: re-score an existing run_ folder against the labelled input instead of "
             "running the pipeline (no cloud processing).",
    ),
) -> None:
    """Run the pipeline on a labelled file and score it against the gold labels."""
    if run_dir is not None:
        _evaluate_existing_run(run_dir, input, id_column, text_column, date_column, sheet)
        return

    result = _run_pipeline_command(
        "evaluate", input, allow_cloud_processing, text_column, id_column, date_column, sheet,
        config_dir, output_dir, quiet, verbose, seed=seed,
    )
    status = result.get("evaluation_results", {}).get("evaluation_summary", {}).get("status")
    typer.echo("Evaluation completed successfully.")
    typer.echo(f"Evaluation folder: {result['run_folder']}")
    typer.echo(f"Evaluation status: {status}")
    if status != "available":
        raise typer.Exit(code=EXIT_EVALUATION)


def _evaluate_existing_run(
    run_dir: Path, inputs: list[Path], id_column, text_column, date_column, sheet,
) -> None:
    """Score saved predictions from an existing run folder (no models, no Gemini)."""
    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
        raise _error("please provide a valid run directory (outputs/run_...).", EXIT_INPUT)

    _check_input_files(inputs, text_column, sheet)

    classified_path = run_dir / "classified_reviews.csv"
    clusters_path = run_dir / "clusters.json"
    metadata_path = run_dir / "run_metadata.json"
    for path in (classified_path, clusters_path, metadata_path):
        if not path.is_file():
            raise _error(f"Required run artifact is missing: {path.name}", EXIT_INPUT)

    try:
        classified_df = pd.read_csv(classified_path, dtype=str, keep_default_na=False)
        clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        missing_columns = {"review_id", "predicted_major_category", "sentiment"} - set(classified_df.columns)
        if missing_columns:
            raise ValueError(f"Prediction file is missing columns: {sorted(missing_columns)}")
        if not isinstance(clusters, dict):
            raise ValueError("clusters.json must contain a JSON object.")

        if isinstance(clusters.get("assignments"), list):
            subcategory_assignments = clusters["assignments"]
        elif "subcategory_id" in classified_df.columns:
            subcategory_assignments = [
                {"review_id": row["review_id"].strip(), "subcategory_id": row["subcategory_id"] or "UNASSIGNED"}
                for row in classified_df.to_dict(orient="records")
            ]
        else:
            raise ValueError("clusters.json must contain an assignments list.")

        from app_review_intelligence.adapters.input.interface import load_and_prepare

        _, evaluation_labels, dataset_metadata = load_and_prepare(
            [str(path) for path in inputs], **_adapter_kwargs(id_column, text_column, date_column, sheet)
        )
        if not evaluation_labels or not any(
            any(record.get(field) not in (None, "") for field in
                ("gold_major_category", "gold_minor_category", "gold_sentiment"))
            for record in evaluation_labels
        ):
            raise ValueError("The input file does not contain usable ground-truth labels.")

        major_predictions, sentiment_results = [], []
        for record in classified_df.to_dict(orient="records"):
            review_id = record["review_id"].strip()
            if not review_id:
                raise ValueError("Prediction file contains an empty review_id.")
            major_predictions.append({
                "review_id": review_id,
                "predicted_major_category": record["predicted_major_category"],
            })
            sentiment_results.append({"review_id": review_id, "sentiment": record["sentiment"]})

        from app_review_intelligence.pipeline.interface import create_run_folder

        eval_folder = create_run_folder("eval", run_dir.parent)

        from app_review_intelligence.evaluation.interface import evaluate_predictions

        results = evaluate_predictions(
            major_predictions=major_predictions,
            sentiment_results=sentiment_results,
            subcategory_assignments=subcategory_assignments,
            evaluation_labels=evaluation_labels,
            output_dir=eval_folder,
            run_metadata={
                **run_metadata,
                "evaluated_run_id": run_dir.name,
                "evaluation_input_metadata": {
                    key: value for key, value in dataset_metadata.items()
                    if key not in ("record_sources", "preprocessing")
                },
            },
            input_file=inputs[0],
            seed=run_metadata.get("random_seed", run_metadata.get("seed", 42)),
            model_ids=run_metadata.get("model_ids"),
        )
    except Exception as exc:
        raise _error(f"Evaluation failed: {exc}", EXIT_GENERAL)

    typer.echo("Evaluation completed successfully.")
    typer.echo(f"Evaluation folder: {eval_folder}")
    typer.echo(f"Evaluation status: {results['evaluation_summary']['status']}")


@app.command("download-models", epilog=EXIT_CODE_HELP)
def download_models(
    force: bool = typer.Option(False, "--force", help="Download again even if already present."),
) -> None:
    """Download the local models (MiniLM, CardiffNLP) into models/ for offline use."""
    from app_review_intelligence.model_store.interface import download_all_models, model_revision

    try:
        paths = download_all_models(force)
    except Exception as exc:
        raise _error(str(exc), EXIT_MODEL)
    for model_id, path in paths.items():
        typer.echo(f"{model_id} -> {path} (revision {model_revision(model_id)})")
    typer.echo("Gemini runs through the API and is not downloaded.")


def main() -> None:
    """Entry point used by inference.py."""
    app()
