"""
Tests for pipeline (Person 6).

The pipeline only connects components through their interface.py functions
(writing_block.md sections 1 and 3, Docs/02 section 4).
"""

import ast

import pytest

from conftest import PACKAGE_DIR, get_component_function


RUN_PIPELINE = ("app_review_intelligence.pipeline.interface", "run_pipeline")


def test_pipeline_function_exists():
    run_pipeline = get_component_function(*RUN_PIPELINE)

    assert callable(run_pipeline)


def test_pipeline_rejects_missing_input():
    run_pipeline = get_component_function(*RUN_PIPELINE)

    with pytest.raises((ValueError, FileNotFoundError)):
        run_pipeline([], {})


def test_pipeline_rejects_nonexistent_file(tmp_path):
    run_pipeline = get_component_function(*RUN_PIPELINE)

    with pytest.raises(FileNotFoundError):
        run_pipeline([str(tmp_path / "does_not_exist.csv")], {})


def test_pipeline_imports_components_only_through_interfaces():
    get_component_function(*RUN_PIPELINE)
    source = PACKAGE_DIR / "pipeline" / "implementation.py"
    if not source.exists():
        pytest.skip("pipeline/implementation.py is not implemented yet")

    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            if module.startswith("app_review_intelligence.") and not module.startswith(
                "app_review_intelligence.pipeline"
            ):
                assert module.endswith(".interface"), f"pipeline imports {module}, not an interface"
