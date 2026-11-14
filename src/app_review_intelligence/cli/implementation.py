import json
from pathlib import Path

import pandas as pd
import typer

from app_review_intelligence.pipeline.interface import run_pipeline

app = typer.Typer(
    help="App Review Intelligence command-line interface.",
    no_args_is_help=True,
)

INVALID_INPUT_EXIT_CODE = 3


def validate_input_file(input_path: Path) -> None:
    """Validate the input file without creating output folders."""
    if not input_path.is_file():
        typer.echo(f"Input file does not exist: {input_path}", err=True)
        raise typer.Exit(code=INVALID_INPUT_EXIT_CODE)

    suffix = input_path.suffix.lower()

    if suffix not in {".csv", ".xlsx"}:
        typer.echo(
            "Unsupported file format. Use CSV or XLSX.",
            err=True,
        )
        raise typer.Exit(code=INVALID_INPUT_EXIT_CODE)

    try:
        if suffix == ".csv":
            dataframe = pd.read_csv(input_path, nrows=5)
        else:
            dataframe = pd.read_excel(input_path, nrows=5)
    except Exception as exc:
        typer.echo(f"Could not read input file: {exc}", err=True)
        raise typer.Exit(code=INVALID_INPUT_EXIT_CODE)

    if "Review" not in dataframe.columns:
        typer.echo(
            "Input file must contain a 'Review' column.",
            err=True,
        )
        raise typer.Exit(code=INVALID_INPUT_EXIT_CODE)


@app.command()
def validate(
    input: Path = typer.Option(
        ...,
        "--input",
        help="Path to the CSV or XLSX review file.",
        exists=False,
    ),
) -> None:
    """Validate an input review file."""
    validate_input_file(input)
    typer.echo("Input file is valid.")


@app.command()
def run(
    input: Path = typer.Option(
        ...,
        "--input",
        help="Path to the CSV or XLSX review file.",
    ),
    allow_cloud_processing: bool = typer.Option(
        False,
        "--allow-cloud-processing",
        help="Allow review text to be sent to the configured cloud service.",
    ),
) -> None:
    """Run the complete review analysis pipeline."""
    if not allow_cloud_processing:
        typer.echo(
            "Cloud processing requires --allow-cloud-processing.",
            err=True,
        )
        raise typer.Exit(code=2)

    import os

    if not os.environ.get("GEMINI_API_KEY"):
        typer.echo(
            "GEMINI_API_KEY environment variable is required.",
            err=True,
        )
        raise typer.Exit(code=2)

    validate_input_file(input)

    try:
        result = run_pipeline(input)
    except Exception as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Pipeline completed successfully.")
    typer.echo(f"Run folder: {result['run_folder']}")
    typer.echo(f"Report: {result['report_path']}")



@app.command()
def evaluate(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Path to an existing run directory.",
    ),
    input: Path = typer.Option(
        ...,
        "--input",
        help="Original labelled CSV or XLSX input file.",
    ),
) -> None:
    """Evaluate predictions from an existing run against gold labels."""

    # 1. Validate paths.
    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
        typer.echo(
            "Please provide a valid run directory.",
            err=True,
        )
        raise typer.Exit(code=INVALID_INPUT_EXIT_CODE)

    validate_input_file(input)

    classified_path = run_dir / "classified_reviews.csv"
    clusters_path = run_dir / "clusters.json"
    metadata_path = run_dir / "run_metadata.json"

    for path in (classified_path, clusters_path, metadata_path):
        if not path.is_file():
            typer.echo(
                f"Required run artifact is missing: {path.name}",
                err=True,
            )
            raise typer.Exit(code=INVALID_INPUT_EXIT_CODE)

    try:
        # 2. Load saved prediction artifacts.
        classified_df = pd.read_csv(
            classified_path,
            dtype=str,
            keep_default_na=False,
        )

        with clusters_path.open("r", encoding="utf-8") as file:
            clusters = json.load(file)

        with metadata_path.open("r", encoding="utf-8") as file:
            run_metadata = json.load(file)

        required_columns = {
            "review_id",
            "predicted_major_category",
            "sentiment",
        }

        missing_columns = required_columns - set(classified_df.columns)

        if missing_columns:
            raise ValueError(
                f"Prediction file is missing columns: {sorted(missing_columns)}"
            )

        if not isinstance(clusters, dict):
            raise ValueError("clusters.json must contain a JSON object.")

        subcategory_assignments = clusters.get("assignments")

        if not isinstance(subcategory_assignments, list):
            raise ValueError(
                "clusters.json must contain an assignments list."
            )

        # 3. Load original labels separately.
        from app_review_intelligence.adapters.input.interface import (
            load_and_prepare,
        )

        _, evaluation_labels, dataset_metadata = load_and_prepare(
            [str(input)]
        )

        if not evaluation_labels or not any(
            any(
                record.get(field) not in (None, "")
                for field in (
                    "gold_major_category",
                    "gold_minor_category",
                    "gold_sentiment",
                )
            )
            for record in evaluation_labels
        ):
            raise ValueError(
                "The input file does not contain usable ground-truth labels."
            )

        # 4. Convert saved predictions into evaluator input.
        major_predictions = []
        sentiment_results = []

        for record in classified_df.to_dict(orient="records"):
            review_id = record["review_id"].strip()

            if not review_id:
                raise ValueError(
                    "Prediction file contains an empty review_id."
                )

            major_predictions.append({
                "review_id": review_id,
                "predicted_major_category": (
                    record["predicted_major_category"]
                ),
            })

            sentiment_results.append({
                "review_id": review_id,
                "sentiment": record["sentiment"],
            })

        # 5. Create a new, unique evaluation directory.
        from app_review_intelligence.pipeline.interface import (create_run_folder,
        )

        eval_folder = create_run_folder(
            "eval",
            run_dir.parent,
        )

        # 6. Run the existing evaluator.
        from app_review_intelligence.evaluation.interface import (
            evaluate_predictions,
        )

        results = evaluate_predictions(
            major_predictions=major_predictions,
            sentiment_results=sentiment_results,
            subcategory_assignments=subcategory_assignments,
            evaluation_labels=evaluation_labels,
            output_dir=eval_folder,
            run_metadata={
                **run_metadata,
                "evaluation_input_metadata": dataset_metadata,
            },
            input_file=input,
            seed=run_metadata.get("seed", 42),
            model_ids=run_metadata.get("model_ids"),
        )

    except Exception as exc:
        typer.echo(
            f"Evaluation failed: {exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("Evaluation completed successfully.")
    typer.echo(f"Evaluation folder: {eval_folder}")
    typer.echo(
        f"Evaluation status: "
        f"{results['evaluation_summary']['status']}"
    )
