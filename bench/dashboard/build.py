"""Build data.json from a pytest-benchmark JSON export.

Consumed by the ``Bench Dashboard`` GitHub Actions workflow. Takes the raw
``bench-current.json`` produced by ``pytest bench/ --benchmark-json=...`` and
produces a smaller, dashboard-friendly ``data.json`` alongside the static
``index.html`` / ``styles.css`` / ``dashboard.js`` bundle.

Usage (from repo root):

    python bench/dashboard/build.py \\
        --input  bench-current.json \\
        --output site/data.json \\
        --commit "$(git rev-parse --short HEAD)" \\
        --version "$(python -c 'import mcp_server; print(mcp_server.__version__)')"

The output schema is intentionally flat — ``dashboard.js`` handles the rest
(categorization, sorting, filtering, chart rendering).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _short_name(full: str) -> str:
    """Trim ``tests/test_bench_search.py::TestFoo::test_bench_bar`` → ``test_bench_bar``.

    The pytest-benchmark ``name`` field is the full nodeid. The dashboard is
    much easier to read with only the terminal test function name, so we drop
    the path + class prefix but keep the parametrization suffix if present.
    """
    core = full.rsplit("::", 1)[-1]
    return core


def _pct_or_none(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator`` guarded against zero."""
    try:
        if not denominator:
            return None
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return None


def build_payload(
    raw: dict[str, Any],
    commit: str,
    version: str,
) -> dict[str, Any]:
    """Transform pytest-benchmark output into the dashboard schema.

    Args:
        raw: Parsed contents of ``bench-current.json``.
        commit: Short git SHA of the commit that produced this run.
        version: knowledge-rag package version at the time of the run.

    Returns:
        A dict ready to serialize as ``data.json``.
    """
    rows: list[dict[str, Any]] = []
    for b in raw.get("benchmarks", []):
        stats = b.get("stats", {}) or {}
        median = float(stats.get("median", 0.0))
        stddev = float(stats.get("stddev", 0.0))
        ops = float(stats.get("ops", 0.0))
        rows.append(
            {
                "name": _short_name(b.get("name") or b.get("fullname") or "?"),
                "median_ns": median * 1_000_000_000,
                "stddev_ns": stddev * 1_000_000_000,
                "ops": ops,
                "rounds": int(stats.get("rounds", 0)),
                "iqr_ratio": _pct_or_none(
                    float(stats.get("iqr", 0.0)) * 1_000_000_000,
                    median * 1_000_000_000,
                ),
            }
        )
    rows.sort(key=lambda r: r["median_ns"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit or "unknown",
        "version": version or "unknown",
        "results": rows,
    }


def main() -> int:
    """CLI entry point — parse args, read input, write output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, type=Path, help="pytest-benchmark JSON export")
    parser.add_argument("--output", "-o", required=True, type=Path, help="destination data.json path")
    parser.add_argument("--commit", default="", help="short git SHA")
    parser.add_argument("--version", default="", help="package version")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        return 2

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    payload = build_payload(raw, commit=args.commit, version=args.version)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote {args.output} — {len(payload['results'])} benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
