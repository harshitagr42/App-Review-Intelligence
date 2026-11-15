"""App Review Intelligence entry point: `python inference.py --help`."""

import sys
from pathlib import Path

# Run from a source checkout without installing the package.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app_review_intelligence.cli.interface import main  # noqa: E402


if __name__ == "__main__":
    main()
