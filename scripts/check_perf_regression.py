"""Compare two pytest-benchmark JSON outputs and fail on >10% regression.

Pillar 5 — Scalability: every PR runs ``bench/`` against master AND
against the branch. This script ingests both result files and emits a
regression report. Median is the comparison metric (least sensitive to
outlier samples).

A benchmark regresses when its median wall time or measured RSS bytes grow by more than
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
import math
import os
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD = 0.10  # 10%
BYPASS_LABEL = "skip-perf-gate"


def _load(path: Path) -> dict[str, dict[str, Any]]:
    """Return explicit units; legacy memory timings are not RSS measurements."""
    if not path.exists():
        print(f"[ERROR] Benchmark file missing: {path}", file=sys.stderr)
        raise SystemExit(2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for bench in payload.get("benchmarks", []):
        out[bench["name"]] = _measurement(bench)
    if not out:
        print(f"[ERROR] No benchmarks found in {path}", file=sys.stderr)
        raise SystemExit(2)
    return out


def _measurement(bench: dict[str, Any]) -> dict[str, Any]:
    """Read RSS bytes from extra_info without relabelling pytest timing stats."""
    extra = bench.get("extra_info") or {}
    name = bench.get("fullname", bench["name"]).replace("\\", "/")
    legacy_memory = "test_bench_memory.py" in name or bench["name"] in {
        "test_bench_orchestrator_idle_rss",
        "test_bench_query_cache_5000_entries",
    }
    result = {
        "median": bench["stats"]["median"],
        "unit": "seconds",
        "workload_version": extra.get("workload_version", 1),
    }
    if extra.get("measurement") == "rss_peak_delta_bytes":
        result.update(median=extra.get("rss_peak_delta_bytes"), unit="bytes")
    elif legacy_memory:
        result.update(median=None, unit="bytes", reason="legacy artifact has no RSS byte measurement")
    value = result["median"]
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
        raise ValueError(f"Invalid {result['unit']} measurement for {bench['name']}: {value!r}")
    return result


def _incomparable(master: dict[str, Any], branch: dict[str, Any]) -> str | None:
    """Explain why two rows do not describe the same measurement/workload."""
    if master["median"] is None or branch["median"] is None:
        return str(master.get("reason") or branch.get("reason"))
    if master["unit"] != branch["unit"]:
        return f"different units: {master['unit']} / {branch['unit']}"
    if master["workload_version"] != branch["workload_version"]:
        return "workload definition changed; a new baseline is required for this benchmark"
    if master["median"] == 0:
        return "zero baseline: relative change is undefined; absolute test budget still applies"
    return None


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
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and non-negative")

    master = _load(args.master_json)
    branch = _load(args.branch_json)

    common = sorted(set(master) & set(branch))
    only_master = sorted(set(master) - set(branch))
    only_branch = sorted(set(branch) - set(master))

    if only_master:
        print(f"[WARN] Benchmarks present in master but missing from branch: {only_master}", file=sys.stderr)
    if only_branch:
        print(f"[INFO] New benchmarks added by branch: {only_branch}")

    regressions: list[tuple[str, float, float, float, str]] = []
    improvements: list[tuple[str, float]] = []
    compared = 0

    for name in common:
        reason = _incomparable(master[name], branch[name])
        if reason:
            print(f"[INCOMPARABLE] {name}: {reason}")
            continue
        m_median = master[name]["median"]
        b_median = branch[name]["median"]
        compared += 1
        delta = (b_median - m_median) / m_median
        if delta > args.threshold and not math.isclose(delta, args.threshold, rel_tol=1e-12, abs_tol=1e-12):
            regressions.append((name, m_median, b_median, delta, branch[name]["unit"]))
        elif delta < -args.threshold:
            improvements.append((name, delta))

    print(f"\nBenchmarks compared: {compared} ({len(common) - compared} explicitly incomparable)")
    print(f"Threshold: ±{args.threshold * 100:.0f}%\n")

    if improvements:
        print("Improvements (lower latency or memory, no action needed):")
        for name, delta in improvements:
            print(f"  [IMPROVED] {name}  {_format_delta(delta)}")
        print()

    if regressions:
        # Honor the bypass label when set deliberately on a PR
        labels = {label.strip() for label in os.environ.get("PR_LABELS", "").split(",") if label.strip()}
        if BYPASS_LABEL in labels:
            print(f"[WARN] Regressions detected but PR has '{BYPASS_LABEL}' label — bypassing:", file=sys.stderr)
            for name, m, b, delta, unit in regressions:
                print(
                    f"  - {name}  median {m:.6g} -> {b:.6g} {unit} ({_format_delta(delta)})",
                    file=sys.stderr,
                )
            return 0

        print("[FAIL] Performance regressions detected:", file=sys.stderr)
        for name, m, b, delta, unit in regressions:
            print(
                f"  [REGRESSION] {name}  median {m:.6g} -> {b:.6g} {unit} ({_format_delta(delta)})",
                file=sys.stderr,
            )
        print(
            "\nIf this regression is intentional and accepted:\n"
            "  - Document the trade-off in the PR description\n"
            f"  - Apply the '{BYPASS_LABEL}' label to bypass\n",
            file=sys.stderr,
        )
        return 1

    if not compared:
        print("[ERROR] No comparable measurements; cannot establish a regression verdict.", file=sys.stderr)
        return 2
    print("[OK] No comparable benchmarks regressed beyond threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
