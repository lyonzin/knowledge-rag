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

# Since README v5 (PR #181), the release history was migrated out of README.md
# into a top-level CHANGELOG.md. Keep README.md as a legacy fallback so this
# script keeps working during the migration window and on branches that
# predate the migration.
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
README = REPO_ROOT / "README.md"


def _changelog_file() -> Path:
    """Return the changelog source of truth: CHANGELOG.md if present, else README.md."""
    return CHANGELOG if CHANGELOG.exists() else README


# Types whose PRs MUST update CHANGELOG (= ## Unreleased section)
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


def _read_versioned_sections(text: str) -> str:
    """Extract all versioned headings (### v3.X.Y) that don't exist in base."""
    pattern = re.compile(
        r"^### v\d+\.\d+\.\d+[^\n]*\n(?P<body>.*?)(?=^### |^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    return "\n".join(m.group("body") for m in pattern.finditer(text))


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

    # User-facing change — ensure ## Unreleased gained at least one bullet.
    # Read from CHANGELOG.md when present (post-README-v5), else README.md legacy.
    changelog_file = _changelog_file()
    if not changelog_file.exists():
        print(f"[ERROR] neither CHANGELOG.md nor README.md found at {REPO_ROOT}", file=sys.stderr)
        return 2

    current = changelog_file.read_text(encoding="utf-8")
    rel_path = changelog_file.relative_to(REPO_ROOT).as_posix()
    try:
        base = _git("show", f"{args.base_ref}:{rel_path}")
    except subprocess.CalledProcessError:
        # File did not exist at the base ref (e.g. CHANGELOG.md is brand-new).
        # Fall back to README.md at the base ref for the migration window.
        try:
            base = _git("show", f"{args.base_ref}:README.md")
            print(f"[INFO] {rel_path} did not exist at {args.base_ref}; diffing against README.md instead.")
        except subprocess.CalledProcessError:
            print(f"[WARN] Could not read {rel_path} or README.md at {args.base_ref}. Skipping diff check.")
            return 0

    unreleased_current = _bullet_count(_read_unreleased_section(current))
    unreleased_base = _bullet_count(_read_unreleased_section(base))
    versioned_current = _bullet_count(_read_versioned_sections(current))
    versioned_base = _bullet_count(_read_versioned_sections(base))

    unreleased_gained = unreleased_current > unreleased_base
    versioned_gained = versioned_current > versioned_base

    if not unreleased_gained and not versioned_gained:
        print(
            "[FAIL] User-facing PR ({type}) but {file} changelog has not gained any new bullet.\n"
            "       Add an entry under '### Unreleased' or a versioned '### v3.X.Y' heading,\n"
            "       or apply the 'skip-changelog' label if truly not needed.\n"
            "       Unreleased: {ucur} (base {ubase}), Versioned: {vcur} (base {vbase})".format(
                type=commit_type,
                file=rel_path,
                ucur=unreleased_current,
                ubase=unreleased_base,
                vcur=versioned_current,
                vbase=versioned_base,
            ),
            file=sys.stderr,
        )
        return 1

    where = "Unreleased" if unreleased_gained else "versioned heading"
    print(f"[OK] CHANGELOG updated under {where} (type: {commit_type})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
