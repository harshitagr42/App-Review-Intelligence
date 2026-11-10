
from conftest import get_component_function


def test_discovery_returns_expected_structure(sample_records):
    discover = get_component_function(
        "app_review_intelligence.components.subcategory.interface",
        "discover_subcategories",
    )

    records = [
        {
            **record,
            "predicted_major_category": "Performance and Stability",
        }
        for record in sample_records
    ]

    result = discover(records)

    assert isinstance(result, dict)
    assert "subcategories" in result
    assert "assignments" in result

    assert isinstance(result["subcategories"], list)
    assert isinstance(result["assignments"], list)


def test_subcategory_assignments_use_valid_review_ids(sample_records):
    discover = get_component_function(
        "app_review_intelligence.components.subcategory.interface",
        "discover_subcategories",
    )

    records = [
        {
            **record,
            "predicted_major_category": "Performance and Stability",
        }
        for record in sample_records
    ]

    result = discover(records)

    input_ids = {r["review_id"] for r in records}
    assigned_ids = [
        assignment["review_id"]
        for assignment in result["assignments"]
    ]

    assert set(assigned_ids).issubset(input_ids)
    assert len(assigned_ids) == len(set(assigned_ids))


def test_every_review_is_assigned_or_explicitly_unassigned(sample_records):
    discover = get_component_function(
        "app_review_intelligence.components.subcategory.interface",
        "discover_subcategories",
    )

    records = [
        {
            **record,
            "predicted_major_category": "Performance and Stability",
        }
        for record in sample_records
    ]

    result = discover(records)

    input_ids = {r["review_id"] for r in records}
    assigned_ids = {
        assignment["review_id"]
        for assignment in result["assignments"]
    }

    assert input_ids == assigned_ids


def test_assignments_reference_existing_subcategories(sample_records):
    discover = get_component_function(
        "app_review_intelligence.components.subcategory.interface",
        "discover_subcategories",
    )

    records = [
        {
            **record,
            "predicted_major_category": "Performance and Stability",
        }
        for record in sample_records
    ]

    result = discover(records)

    valid_subcategory_ids = {
        category["subcategory_id"]
        for category in result["subcategories"]
    }

    for assignment in result["assignments"]:
        subcategory_id = assignment["subcategory_id"]

        assert (
            subcategory_id == "UNASSIGNED"
            or subcategory_id in valid_subcategory_ids
        )
