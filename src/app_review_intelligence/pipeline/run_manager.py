
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_run_folder(
    folder_type: str,
    output_root: str | Path = "outputs",
) -> Path:
    """
    Create a unique run or evaluation folder.
    """
    if folder_type not in {"run", "eval"}:
        raise ValueError("folder_type must be 'run' or 'eval'")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for _ in range(10):
        unique_id = uuid4().hex[:8]
        folder_name = f"{folder_type}_{timestamp}_{unique_id}"
        folder_path = root / folder_name

        try:
            folder_path.mkdir(exist_ok=False)
            return folder_path
        except FileExistsError:
            continue

    raise FileExistsError(
        f"Could not create a unique {folder_type} folder after 10 attempts."
    )
