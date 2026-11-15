
from app_review_intelligence.pipeline.run_manager import create_run_folder


def test_creates_run_folder(tmp_path):
    folder = create_run_folder("run", tmp_path)

    assert folder.exists()
    assert folder.is_dir()
    assert folder.name.startswith("run_")


def test_creates_evaluation_folder(tmp_path):
    folder = create_run_folder("eval", tmp_path)

    assert folder.exists()
    assert folder.is_dir()
    assert folder.name.startswith("eval_")


def test_creates_unique_folders(tmp_path):
    first = create_run_folder("run", tmp_path)
    second = create_run_folder("run", tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_rejects_invalid_folder_type(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        create_run_folder("invalid", tmp_path)
