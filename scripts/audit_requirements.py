"""Export the complete dependency closure from a pip installation report.

Resolve first with ``python -m pip install --dry-run --ignore-installed
--report deps-report.json .``. This avoids pip-tools' dependency on private
pip APIs and excludes only the unpublished local project from pip-audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name
from packaging.version import Version


def dependency_pins(report: dict[str, Any], project: str = "knowledge-rag") -> list[str]:
    """Return validated, exact pins for every resolved dependency."""
    if report.get("version") != "1" or not isinstance(report.get("install"), list):
        raise ValueError("Expected a pip installation report with version 1 and an install list")
    excluded = canonicalize_name(project)
    versions: dict[str, str] = {}
    for distribution in report["install"]:
        metadata = distribution["metadata"]
        name = canonicalize_name(metadata["name"], validate=True)
        if name == excluded:
            continue
        version = str(Version(metadata["version"]))
        if name in versions and versions[name] != version:
            raise ValueError(f"Conflicting resolved versions for {name}")
        versions[name] = version
    if not versions:
        raise ValueError("No dependencies found; refusing to audit an empty requirement set")
    return [f"{name}=={version}" for name, version in sorted(versions.items())]


def main() -> int:
    """Parse a pip report and write the requirements consumed by pip-audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pins = dependency_pins(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.write_text("\n".join(pins) + "\n", encoding="utf-8")
    print(f"Exported {len(pins)} resolved dependency pins to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
