#!/usr/bin/env python3
"""PreToolUse guard for Bash commands: enforces the project's absolute git rules.

Blocked (exit code 2, reason on stderr):
  1. `git add` / `git commit` while on `main` or `development`, or on any branch
     whose name does not start with `feature/`, `fix/` or `bug/`.
  2. Creating a branch whose name does not start with `feature/`, `fix/` or `bug/`.
  3. Any merge: `git merge`, `git pull` (unless --rebase / --ff-only), `gh pr merge`.
  4. Deleting branches: `git branch -d/-D/--delete`, `git push --delete`,
     `git push <remote> :<branch>`, `gh api ... -X DELETE .../git/refs/...`.
  5. Pushing to `main` / `development`, `git push --all/--mirror/--force` to them.
  6. Committing or adding anything under `Docs/` or `dataset/`, or `.env` / key files.

Only stdlib is used so this runs anywhere `python3` exists.
"""

import json
import re
import shlex
import subprocess
import sys

PROTECTED_BRANCHES = {"main", "development"}
ALLOWED_PREFIXES = ("feature/", "fix/", "bug/")
FORBIDDEN_PATH_RE = re.compile(
    r"(^|/)(Docs|dataset)(/|$)|(^|/)\.env($|\.)|\.(pem|key|p12|pfx)$"
)
SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n")


def block(reason):
    sys.stderr.write(f"BLOCKED by .claude/hooks/guard_git.py: {reason}\n")
    sys.exit(2)


def run_git(args, cwd=None):
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def branch_allowed(name):
    return name.startswith(ALLOWED_PREFIXES) and name not in PROTECTED_BRANCHES


def tokenize(segment):
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def strip_git_globals(tokens):
    """Return (subcommand_tokens, cwd) for a `git ...` token list."""
    i = 1
    cwd = None
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] == "-C" and i + 1 < len(tokens):
            cwd = tokens[i + 1]
            i += 2
            continue
        if tokens[i] in ("-c", "--git-dir", "--work-tree") and i + 1 < len(tokens):
            i += 2
            continue
        i += 1
    return tokens[i:], cwd


def check_paths(paths, action):
    bad = [p for p in paths if FORBIDDEN_PATH_RE.search(p)]
    if bad:
        block(
            f"{action} would include protected paths {bad}. "
            "Docs/, dataset/, .env and key files must never be committed."
        )


def check_git(sub, cwd):
    if not sub:
        return
    cmd, args = sub[0], sub[1:]
    current = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)

    if cmd in ("add", "commit"):
        if current and not branch_allowed(current):
            block(
                f"`git {cmd}` on branch '{current}'. Work only on branches starting "
                f"with {', '.join(ALLOWED_PREFIXES)}; never on main/development."
            )
        if cmd == "add":
            if any(a in ("-A", "--all", ".", "-f", "--force") for a in args):
                # Broad add: check what would be picked up, including ignored if forced.
                listed = run_git(["status", "--porcelain", "--untracked-files=all"], cwd)
                candidates = [line[3:] for line in listed.splitlines()]
                if any(a in ("-f", "--force") for a in args):
                    candidates += [a for a in args if not a.startswith("-")]
                check_paths(candidates, "`git add`")
            check_paths([a for a in args if not a.startswith("-")], "`git add`")
        if cmd == "commit":
            staged = run_git(["diff", "--cached", "--name-only"], cwd).splitlines()
            if any(a == "--all" or (re.fullmatch(r"-[A-Za-z]+", a) and "a" in a) for a in args):
                staged += run_git(["diff", "--name-only"], cwd).splitlines()
            check_paths(staged, "`git commit`")
        return

    if cmd == "merge":
        block("`git merge` is forbidden. Merging is done by humans via pull requests.")

    if cmd == "pull" and not any(a in ("--rebase", "-r", "--ff-only") for a in args):
        block("`git pull` without --rebase/--ff-only creates merges, which are forbidden.")

    if cmd == "branch":
        if any(a == "--delete" or (re.fullmatch(r"-[A-Za-z]+", a) and ("d" in a or "D" in a)) for a in args):
            block("Deleting branches is forbidden.")
        if any(a in ("-m", "-M", "--move") for a in args):
            block("Renaming branches is forbidden (could rename main/development).")
        names = [a for a in args if not a.startswith("-")]
        listing_flags = {"-a", "-r", "-l", "--list", "-v", "-vv", "--all", "--remotes",
                         "--show-current", "--contains", "--merged", "--no-merged"}
        if names and not any(a in listing_flags for a in args) and not branch_allowed(names[0]):
            block(f"New branch '{names[0]}' must start with {', '.join(ALLOWED_PREFIXES)}.")

    if cmd in ("checkout", "switch"):
        for flag in ("-b", "-B", "-c", "-C", "--create", "--force-create", "--orphan"):
            if flag in args:
                idx = args.index(flag)
                if idx + 1 < len(args) and not branch_allowed(args[idx + 1]):
                    block(
                        f"New branch '{args[idx + 1]}' must start with "
                        f"{', '.join(ALLOWED_PREFIXES)}."
                    )

    if cmd == "push":
        if any(a in ("-d", "--delete") for a in args):
            block("Deleting remote branches is forbidden.")
        if any(a in ("--all", "--mirror", "--prune") for a in args):
            block("`git push --all/--mirror/--prune` is forbidden.")
        positional = [a for a in args if not a.startswith("-")]
        for ref in positional[1:]:
            if ref.startswith(":"):
                block("Deleting remote branches (`push :branch`) is forbidden.")
            dest = ref.split(":")[-1].lstrip("+")
            dest = dest.removeprefix("refs/heads/")
            if dest in PROTECTED_BRANCHES:
                block(f"Pushing to '{dest}' is forbidden.")
        if len(positional) <= 1 and current in PROTECTED_BRANCHES:
            block(f"Pushing while on '{current}' is forbidden.")

    if cmd == "update-ref" and "-d" in args:
        block("Deleting refs is forbidden.")


def check_gh(tokens):
    if len(tokens) >= 3 and tokens[1] == "pr" and tokens[2] == "merge":
        block("`gh pr merge` is forbidden. Merging is done by humans.")
    joined = " ".join(tokens)
    if tokens[1:2] == ["api"] and re.search(r"DELETE", joined) and "git/refs" in joined:
        block("Deleting branches through the GitHub API is forbidden.")


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    command = (payload.get("tool_input") or {}).get("command") or ""
    if "git" not in command and "gh" not in command:
        return
    for segment in SEGMENT_SPLIT_RE.split(command):
        tokens = tokenize(segment.strip())
        # Skip leading env assignments like FOO=bar git ...
        while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
            tokens = tokens[1:]
        if not tokens:
            continue
        exe = tokens[0].rsplit("/", 1)[-1]
        if exe == "git":
            sub, cwd = strip_git_globals(tokens)
            check_git(sub, cwd)
        elif exe == "gh":
            check_gh(tokens)


if __name__ == "__main__":
    main()
