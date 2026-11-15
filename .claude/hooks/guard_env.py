#!/usr/bin/env python3
"""PreToolUse guard: Claude must never read or write the local `.env` file.

`.env` holds real secrets (GEMINI_API_KEY). Blocked (exit code 2):
  - Read / Edit / Write / NotebookEdit / Grep / Glob on a path whose file name
    is `.env` or `.env.<anything>` (except `.env.example` / `.env.template`).
  - Bash commands that mention such a file (cat, grep, sed, cp, echo >, source...).

`.env.example` and `.env.template` stay readable and editable.
Only stdlib is used so this runs anywhere `python3` exists.
"""

import json
import re
import sys

ALLOWED_NAMES = {".env.example", ".env.template"}
# A `.env` path token: start of string or a separator, then `.env`, optionally
# followed by `.<suffix>`, ending at a separator/quote/redirect/end.
ENV_TOKEN_RE = re.compile(r"(?:^|[\s/'\"=<>|;&(`])(\.env(?:\.[\w.-]+)?)(?=$|[\s'\";|&)<>`])")


def block(reason):
    sys.stderr.write(f"BLOCKED by .claude/hooks/guard_env.py: {reason}\n")
    sys.exit(2)


def is_protected_name(name):
    return (name == ".env" or name.startswith(".env.")) and name not in ALLOWED_NAMES


def check_path(path):
    if not path:
        return
    name = str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if is_protected_name(name):
        block(f"access to '{path}' is forbidden. Never read or write .env; use .env.example.")


def check_text(text):
    for match in ENV_TOKEN_RE.finditer(text or ""):
        if is_protected_name(match.group(1)):
            block("commands that touch .env are forbidden. Never read or write .env; use .env.example.")


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    tool = payload.get("tool_name") or ""
    data = payload.get("tool_input") or {}

    if tool == "Bash":
        check_text(data.get("command"))
        return
    for key in ("file_path", "notebook_path", "path"):
        check_path(data.get(key))
    if tool in ("Grep", "Glob"):
        check_text(data.get("pattern"))
        check_text(data.get("glob"))


if __name__ == "__main__":
    main()
