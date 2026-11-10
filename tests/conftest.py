
import importlib
import pytest


@pytest.fixture
def sample_records():
    return [
        {
            "review_id": "R0001",
            "review_text": "The app crashes whenever I try to log in.",
            "review_date": "2026-07-21",
        },
        {
            "review_id": "R0002",
            "review_text": "Excellent app. Very easy to use.",
            "review_date": "2026-07-22",
        },
        {
            "review_id": "R0003",
            "review_text": "The latest update made the app extremely slow.",
            "review_date": "2026-07-23",
        },
    ]


@pytest.fixture
def sample_labels():
    return [
        {
            "review_id": "R0001",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "App Crashes",
            "gold_sentiment": "Negative",
        },
        {
            "review_id": "R0002",
            "gold_major_category": "Usability",
            "gold_minor_category": "Easy to Use",
            "gold_sentiment": "Positive",
        },
        {
            "review_id": "R0003",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "Slow Performance",
            "gold_sentiment": "Negative",
        },
    ]


def get_component_function(module_path, function_name):
    """
    Import a component function.

    If the component or function has not been implemented yet,
    skip its tests instead of failing the entire test suite.

    If the component exists but has a real implementation error,
    the test should fail normally.
    """
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        # Skip only when the requested component or one of its
        # parent packages is missing.
        if error.name and (
            module_path == error.name
            or module_path.startswith(error.name + ".")
        ):
            pytest.skip(f"{module_path} is not implemented yet")
        raise

    function = getattr(module, function_name, None)

    if function is None:
        pytest.skip(
            f"{function_name} has not been implemented in {module_path}"
        )

    return function
