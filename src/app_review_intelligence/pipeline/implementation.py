
from pathlib import Path
from typing import Any
import time
import hashlib

from app_review_intelligence.pipeline.run_manager import create_run_folder


def run_pipeline(
    input_paths: str | Path | list[str | Path],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the complete app-review intelligence pipeline.

    Components are called only through their public interfaces.
    Gold labels are kept separate from prediction inputs.
    """
    start_time = time.perf_counter()
    settings = config or {}

    # 1. Validate input paths before importing other components.
    if isinstance(input_paths, (str, Path)):
        paths = [Path(input_paths)]
    else:
        paths = [Path(path) for path in input_paths]

    if not paths:
        raise ValueError("At least one input file must be provided.")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    

    # 2. Import component interfaces only after input validation.
    from app_review_intelligence.adapters.input.interface import (
        load_and_prepare,
    )
    from app_review_intelligence.components.major_category.interface import (
        classify,
        unload_model,
    )
    from app_review_intelligence.components.sentiment.interface import (
        analyze,
    )
    from app_review_intelligence.components.subcategory.interface import (
        discover_subcategories,
    )
    from app_review_intelligence.components.priority.interface import (
        calculate_priorities,
    )
    from app_review_intelligence.evaluation.interface import (
        evaluate_predictions,
    )
    from app_review_intelligence.reporting.interface import (
        generate_report,
    )

    # 3. Load records and keep gold labels separate.
    model_records, evaluation_labels, dataset_metadata = load_and_prepare(
        [str(path) for path in paths]
    )

    if not model_records:
        raise ValueError("No usable review records were found in the input.")
    output_root = Path(settings.get("output_root", "outputs"))
    run_folder = create_run_folder("run", output_root)
    eval_folder = create_run_folder("eval", output_root)

    # 4. Preprocess prediction records.
    # The input adapter already preprocesses records.
    cleaned_records = model_records
    preprocessing_report = dataset_metadata.get("preprocessing", {})

    if not cleaned_records:
        raise ValueError("No reviews remained after preprocessing.")
    # 5. Predict major categories.
    major_predictions = classify(cleaned_records)

    # Release the major-category model before loading the sentiment model.
    unload_model()

    # 6. Analyze sentiment independently of major-category predictions.
    sentiment_results = analyze(cleaned_records)

    # 7. Add predicted major categories to records for subcategory discovery.
    major_by_id = {
        item["review_id"]: item
        for item in major_predictions
    }

    records_with_major_categories = []

    for record in cleaned_records:
        prediction = major_by_id.get(record["review_id"])

        if prediction is None:
            continue

        records_with_major_categories.append({
            **record,
            "predicted_major_category": prediction[
                "predicted_major_category"
            ],
        })

    # 8. Discover subcategories.
    subcategory_result = discover_subcategories(
        records_with_major_categories
    )

    if not isinstance(subcategory_result, dict):
        raise TypeError(
            "discover_subcategories() must return a dictionary."
        )

    subcategory_assignments = subcategory_result.get(
        "assignments", []
    )

    # 9. Calculate priority scores.
    priority_results = calculate_priorities(
        subcategory_result,
        sentiment_results,
    )

    # 10. Collect excluded records for evaluation and reporting.
    excluded_records = []

    for field in (
        "skipped_empty_text",
        "skipped_invalid_date",
        "duplicates_removed",
    ):
        excluded_records.extend(
            preprocessing_report.get(field, [])
        )

    # 11. Prepare run metadata.
    elapsed_seconds = time.perf_counter() - start_time
    try:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        memory_mb = None

    model_ids = {
        "major_category": "sentence-transformers/all-MiniLM-L6-v2",
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "subcategory": "gemini-3.6-flash",
    }
    input_file_metadata = []

    for path in paths:
        file_hash = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                file_hash.update(chunk)

        input_file_metadata.append({
            "file_name": path.name,
            "sha256": file_hash.hexdigest(),
        })
    run_metadata = {
        "input_file_metadata": input_file_metadata,
        "input_files": [path.name for path in paths],
        "input_row_count": dataset_metadata.get("rows_loaded", 0),
        "accepted_row_count": dataset_metadata.get("rows_accepted", len(cleaned_records)),
        "excluded_row_count": len(excluded_records),
        "seed": settings.get("seed", 42),
        "model_ids": model_ids,
        "runtime_seconds": elapsed_seconds,
        "memory_mb": memory_mb,
        "preprocessing": preprocessing_report,
    }

    # 12. Evaluate predictions. Gold labels are used only here.
    evaluation_results = evaluate_predictions(
        major_predictions=major_predictions,
        sentiment_results=sentiment_results,
        subcategory_assignments=subcategory_assignments,
        evaluation_labels=evaluation_labels,
        output_dir=eval_folder,
        run_metadata=run_metadata,
        priority_results=priority_results,
        excluded_records=excluded_records,
        input_file=paths[0],
        seed=settings.get("seed", 42),
        model_ids=model_ids,
    )

    # 13. Combine predictions into the classified review output.
    sentiment_by_id = {
        item["review_id"]: item
        for item in sentiment_results
    }

    assignment_by_id = {
        item["review_id"]: item
        for item in subcategory_assignments
    }

    subcategory_by_id = {
        item["subcategory_id"]: item
        for item in subcategory_result.get("subcategories", [])
    }

    classified_reviews = []

    for record in cleaned_records:
        review_id = record["review_id"]
        major = major_by_id.get(review_id, {})
        sentiment = sentiment_by_id.get(review_id, {})
        assignment = assignment_by_id.get(review_id, {})

        subcategory_id = assignment.get("subcategory_id")
        subcategory = subcategory_by_id.get(subcategory_id, {})

        classified_reviews.append({
            **record,
            "predicted_major_category": major.get(
                "predicted_major_category"
            ),
            "major_category_confidence": major.get("confidence"),
            "sentiment": sentiment.get("sentiment"),
            "sentiment_confidence": sentiment.get("confidence"),
            "subcategory_id": subcategory_id,
            "subcategory_name": subcategory.get("subcategory_name"),
        })

    # 14. Generate the run report and output artifacts.
    logs = (
        f"Input files: {[path.name for path in paths]}\n"
        f"Input records: {len(model_records)}\n"
        f"Accepted records: {len(cleaned_records)}\n"
        f"Excluded records: {len(excluded_records)}\n"
        f"Runtime: {elapsed_seconds:.2f} seconds\n"
    )

    report_path = generate_report(
        results=evaluation_results,
        output_dir=run_folder,
        classified_reviews=classified_reviews,
        clusters=subcategory_result,
        run_metadata=run_metadata,
        logs=logs,
    )

    return {
        "run_folder": str(run_folder),
        "evaluation_folder": str(eval_folder),
        "report_path": report_path,
        "classified_reviews": classified_reviews,
        "clusters": subcategory_result,
        "priority_results": priority_results,
        "evaluation_results": evaluation_results,
        "run_metadata": run_metadata,
    }
