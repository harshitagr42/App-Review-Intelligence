
from conftest import get_component_function


def test_pipeline_function_exists():
    run_pipeline = get_component_function(
        "app_review_intelligence.pipeline.interface",
        "run_pipeline",
    )

    assert callable(run_pipeline)


def test_pipeline_rejects_missing_input():
    run_pipeline = get_component_function(
        "app_review_intelligence.pipeline.interface",
        "run_pipeline",
    )

    try:
        run_pipeline([], {})
    except (ValueError, FileNotFoundError):
        return

    assert False, "Pipeline should reject an empty input list"
