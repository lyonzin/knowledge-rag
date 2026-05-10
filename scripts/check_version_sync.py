"""Verify all three version declarations agree.

knowledge-rag declares its version in three places that must move atomically:
- pyproject.toml          [project] version
- mcp_server/__init__.py  __version__
- npm/package.json        version

Drift between these caused real headaches before v3.8.0 (init was at 3.5.2,
npm at 3.6.2, pyproject at 3.7.0). This script prevents recurrence by
running as a pre-commit hook and as a CI status check.

Exit codes:
    0  all three versions agree
    1  drift detected
    2  could not parse one or more files

Run manually:
    python scripts/check_version_sync.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PYPROJECT = REPO_ROOT / "pyproject.toml"
INIT_PY = REPO_ROOT / "mcp_server" / "__init__.py"
PACKAGE_JSON = REPO_ROOT / "npm" / "package.json"


def read_pyproject_version() -> str:
    """Extract ``version = "X.Y.Z"`` from the [project] table."""
    text = PYPROJECT.read_text(encoding="utf-8")
    # Match the first `version = "..."` occurrence, which is in [project]
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find version in {PYPROJECT}")
    return match.group(1)


def read_init_version() -> str:
    """Extract ``__version__ = "X.Y.Z"``."""
    text = INIT_PY.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {INIT_PY}")
    return match.group(1)


def read_npm_version() -> str:
    """Extract version from npm package.json."""
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    if "version" not in data:
        raise RuntimeError(f"Could not find version in {PACKAGE_JSON}")
    version = data["version"]
    if not isinstance(version, str):
        raise RuntimeError(f"npm version field has wrong type: {type(version).__name__}")
    return version


def main() -> int:
    try:
        py = read_pyproject_version()
        init = read_init_version()
        npm = read_npm_version()
    except (RuntimeError, json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Could not parse a version file: {exc}", file=sys.stderr)
        return 2

    versions = {
        "pyproject.toml": py,
        "mcp_server/__init__.py": init,
        "npm/package.json": npm,
    }

    unique = set(versions.values())
    if len(unique) == 1:
        only = next(iter(unique))
        print(f"[OK] All three version declarations agree: {only}")
        return 0

    print("[FAIL] Version drift detected. All three must match:", file=sys.stderr)
    for source, value in versions.items():
        print(f"  {source:30}  {value}", file=sys.stderr)
    print(
        "\nFix: bump all three files atomically. The release PR template requires this.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
