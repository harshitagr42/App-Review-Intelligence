
import importlib
import pytest

# Values that really occur in dataset/processed/combined_reviews_output.csv.
# Major Class: 9 topics + "Other" (10 labels). Sentiment is lowercase in the
# file; the input adapter normalises it to these Title Case values.
MAJOR_CATEGORIES = (
    "Performance and Stability",
    "Features and Functionality",
    "Billing and Payments",
    "Authentication and Account Access",
    "User Interface and Experience",
    "Customer Support",
    "Privacy and Security",
    "Notifications",
    "Compatibility",
    "Other",
)
SENTIMENTS = ("Positive", "Negative", "Neutral")


@pytest.fixture
def sample_records():
    return [
        {
            "review_id": "1",
            "review_text": "The app keeps crashing every time I open my timetable. Really frustrating.",
            "review_date": "2026-07-21",
        },
        {
            "review_id": "2",
            "review_text": "Excellent app. Very easy to use.",
            "review_date": "2026-07-22",
        },
        {
            "review_id": "3",
            "review_text": "Since the latest update, loading is extremely slow. Please fix this.",
            "review_date": "2026-07-23",
        },
    ]


@pytest.fixture
def sample_labels():
    return [
        {
            "review_id": "1",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "App Crashes",
            "gold_sentiment": "Negative",
        },
        {
            "review_id": "2",
            "gold_major_category": "Other",
            "gold_minor_category": "General Praise",
            "gold_sentiment": "Positive",
        },
        {
            "review_id": "3",
            "gold_major_category": "Performance and Stability",
            "gold_minor_category": "Slow Loading",
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
