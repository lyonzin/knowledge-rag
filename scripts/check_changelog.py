"""Enforce that user-facing PRs add an entry under README.md > ## Unreleased.

Used by .github/workflows/quality-gate.yml on pull_request events. Runs
locally too:

    python scripts/check_changelog.py

Logic:

1. Determine the PR title (from CLI arg, env var, or git log).
2. Inspect the conventional-commit type prefix:
       feat / fix / perf / refactor      => CHANGELOG entry REQUIRED
       docs / chore / ci / test / build  => skipped (not user-facing)
       style / revert                    => skipped
3. Compare README.md against `master`. If the conventional-commit type
   requires an entry, then `## Unreleased` (or `## v3.X.Y`) must have
   gained at least one new bullet line.

Skip via PR label `skip-changelog` if the maintainer agrees a change
genuinely needs none (e.g. internal refactor split across PRs).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

# Types whose PRs MUST update CHANGELOG (= ## Unreleased section in README)
USER_FACING_TYPES = {"feat", "fix", "perf", "refactor"}

# Types that are exempt
SKIP_TYPES = {"docs", "chore", "ci", "test", "build", "style", "revert"}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _resolve_pr_title(cli_title: str | None) -> str | None:
    if cli_title:
        return cli_title
    if env := os.environ.get("PR_TITLE"):
        return env
    # Fall back to the last commit message subject (useful for local runs)
    try:
        return _git("log", "-1", "--pretty=%s")
    except subprocess.CalledProcessError:
        return None


def _conventional_type(title: str) -> str | None:
    """Extract `feat` from `feat(scope): summary` or None if non-conforming."""
    match = re.match(r"^(?P<type>[a-z]+)(?:\([^)]+\))?(?P<bang>!?):", title)
    if not match:
        return None
    return match.group("type")


def _read_unreleased_section(text: str) -> str:
    """Extract the body of the ## Unreleased heading, up to the next H2."""
    pattern = re.compile(
        r"^### Unreleased\s*\n(?P<body>.*?)(?=^### |^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group("body") if match else ""


def _bullet_count(section: str) -> int:
    return sum(1 for line in section.splitlines() if line.lstrip().startswith("- "))


def _is_skip_label_set() -> bool:
    raw = os.environ.get("PR_LABELS", "")
    labels = {label.strip() for label in raw.split(",") if label.strip()}
    return "skip-changelog" in labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-title", help="Override PR title (otherwise PR_TITLE env or last commit subject)")
    parser.add_argument("--base-ref", default="origin/master", help="Branch to diff against (default: origin/master)")
    args = parser.parse_args()

    title = _resolve_pr_title(args.pr_title)
    if not title:
        print("[ERROR] Could not determine PR title.", file=sys.stderr)
        return 2

    commit_type = _conventional_type(title)
    if commit_type is None:
        print(f"[FAIL] PR title does not follow Conventional Commits: {title!r}", file=sys.stderr)
        print("       Expected format: <type>(<scope>): <summary>", file=sys.stderr)
        return 1

    if _is_skip_label_set():
        print("[OK] Label 'skip-changelog' set — bypassing.")
        return 0

    if commit_type in SKIP_TYPES:
        print(f"[OK] Type '{commit_type}' is exempt from CHANGELOG requirement.")
        return 0

    if commit_type not in USER_FACING_TYPES:
        print(f"[OK] Type '{commit_type}' not classified as user-facing (no CHANGELOG required).")
        return 0

    # User-facing change — ensure ## Unreleased gained at least one bullet
    if not README.exists():
        print(f"[ERROR] README.md not found at {README}", file=sys.stderr)
        return 2

    current = README.read_text(encoding="utf-8")
    try:
        base = _git("show", f"{args.base_ref}:README.md")
    except subprocess.CalledProcessError:
        print(f"[WARN] Could not read README.md at {args.base_ref} (first commit?). Skipping diff check.")
        return 0

    current_bullets = _bullet_count(_read_unreleased_section(current))
    base_bullets = _bullet_count(_read_unreleased_section(base))

    if current_bullets <= base_bullets:
        print(
            "[FAIL] User-facing PR ({type}) but README.md '## Unreleased' has not gained any new bullet.\n"
            "       Add a single-line entry, or apply the 'skip-changelog' label if truly not needed.\n"
            "       Current bullets: {cur}, base bullets: {base}".format(
                type=commit_type, cur=current_bullets, base=base_bullets
            ),
            file=sys.stderr,
        )
        return 1

    print(
        f"[OK] CHANGELOG updated: {base_bullets} -> {current_bullets} bullet(s) under '## Unreleased' "
        f"(type: {commit_type})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
