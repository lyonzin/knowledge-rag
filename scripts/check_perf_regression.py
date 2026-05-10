"""Compare two pytest-benchmark JSON outputs and fail on >10% regression.

Pillar 5 — Scalability: every PR runs ``bench/`` against master AND
against the branch. This script ingests both result files and emits a
regression report. Median is the comparison metric (least sensitive to
outlier samples).

A benchmark regresses when its median wall time grows by more than
``REGRESSION_THRESHOLD`` (10% by default). Improvements (faster) are
celebrated, never blocking.

Run locally:
    pytest bench/ --benchmark-json=branch.json
    git stash; git checkout master
    pytest bench/ --benchmark-json=master.json
    git checkout -; git stash pop
    python scripts/check_perf_regression.py master.json branch.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD = 0.10  # 10%
BYPASS_LABEL = "skip-perf-gate"


def _load(path: Path) -> dict[str, dict[str, Any]]:
    """Return mapping of bench name -> stats dict."""
    if not path.exists():
        print(f"[ERROR] Benchmark file missing: {path}", file=sys.stderr)
        raise SystemExit(2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for bench in payload.get("benchmarks", []):
        out[bench["name"]] = bench["stats"]
    if not out:
        print(f"[ERROR] No benchmarks found in {path}", file=sys.stderr)
        raise SystemExit(2)
    return out


def _format_delta(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master_json", type=Path, help="benchmark JSON from master")
    parser.add_argument("branch_json", type=Path, help="benchmark JSON from PR branch")
    parser.add_argument(
        "--threshold",
        type=float,
        default=REGRESSION_THRESHOLD,
        help=f"Fractional regression that fails the gate (default: {REGRESSION_THRESHOLD})",
    )
    args = parser.parse_args()

    master = _load(args.master_json)
    branch = _load(args.branch_json)

    common = sorted(set(master) & set(branch))
    only_master = sorted(set(master) - set(branch))
    only_branch = sorted(set(branch) - set(master))

    if only_master:
        print(f"[WARN] Benchmarks present in master but missing from branch: {only_master}", file=sys.stderr)
    if only_branch:
        print(f"[INFO] New benchmarks added by branch: {only_branch}")

    regressions: list[tuple[str, float, float, float]] = []
    improvements: list[tuple[str, float]] = []

    for name in common:
        m_median = master[name]["median"]
        b_median = branch[name]["median"]
        if m_median <= 0:
            continue
        delta = (b_median - m_median) / m_median
        if delta > args.threshold:
            regressions.append((name, m_median, b_median, delta))
        elif delta < -args.threshold:
            improvements.append((name, delta))

    print(f"\nBenchmarks compared: {len(common)}")
    print(f"Threshold: ±{args.threshold * 100:.0f}%\n")

    if improvements:
        print("Improvements (faster, no action needed):")
        for name, delta in improvements:
            print(f"  ✓ {name}  {_format_delta(delta)}")
        print()

    if regressions:
        # Honor the bypass label when set deliberately on a PR
        labels = {label.strip() for label in os.environ.get("PR_LABELS", "").split(",") if label.strip()}
        if BYPASS_LABEL in labels:
            print(f"[WARN] Regressions detected but PR has '{BYPASS_LABEL}' label — bypassing:", file=sys.stderr)
            for name, m, b, delta in regressions:
                print(
                    f"  - {name}  median {m:.2f} -> {b:.2f}  ({_format_delta(delta)})",
                    file=sys.stderr,
                )
            return 0

        print("[FAIL] Performance regressions detected:", file=sys.stderr)
        for name, m, b, delta in regressions:
            print(
                f"  ✗ {name}  median {m:.2f} -> {b:.2f}  ({_format_delta(delta)})",
                file=sys.stderr,
            )
        print(
            "\nIf this regression is intentional and accepted:\n"
            "  - Document the trade-off in the PR description\n"
            f"  - Apply the '{BYPASS_LABEL}' label to bypass\n",
            file=sys.stderr,
        )
        return 1

    print("[OK] No benchmarks regressed beyond threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
