"""Public interface for the CLI. No logic lives here.

Example:
    from app_review_intelligence.cli.interface import main

    main()   # parses sys.argv: validate | run | evaluate
"""

from app_review_intelligence.cli.implementation import (
    load_dotenv_file as _load_dotenv_file,
    main as _main,
)


def main():
    """Runs the Typer CLI (python inference.py ...)."""
    _main()


def load_dotenv_file():
    """Loads .env into the environment without overriding existing variables; returns the names set."""
    return _load_dotenv_file()
