"""Compare interleaved benchmark runs of master and the PR branch; fail on a consistent >10% regression.

Pillar 5 — Scalability: the perf-bench job runs ``bench/`` four times, in the
order branch, master, master, branch, and writes one pytest-benchmark JSON per
run. Each run gives every benchmark one median: wall time, or tracemalloc peak
bytes for the memory benchmarks. Sub-100 µs medians move by ±15–19% between
runs of identical code, so one median per side cannot tell noise from a change.

A benchmark regresses only when its lowest branch median is above its highest
master median by more than ``REGRESSION_THRESHOLD`` (10% by default). Spreads
that overlap are noise and pass; improvements are reported, never blocking.

Run locally, with master checked out as a worktree in ../master:
    pytest bench/ --benchmark-only --benchmark-json=branch-1.json
    (cd ../master; PYTHONPATH="$PWD" python -m pytest bench/ --benchmark-only --benchmark-json="$OLDPWD/master-1.json")
    # ...the same for master-2.json, then branch-2.json...
    python scripts/check_perf_regression.py --master master-1.json master-2.json --branch branch-1.json branch-2.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD = 0.10  # 10%
BYPASS_LABEL = "skip-perf-gate"
BYTES_MEASUREMENT = "tracemalloc_peak_bytes"
LEGACY_MEMORY_BENCHMARKS = {"test_bench_orchestrator_idle_rss", "test_bench_query_cache_5000_entries"}
# An exact 10% change computes as 0.10000000000000009; it must not count as above the threshold.
FLOAT_TOLERANCE = 1e-12
REGRESSED, IMPROVED, UNCHANGED, NEW, REMOVED = "regressed", "improved", "unchanged", "new", "removed"


def _load(path: Path) -> dict[str, dict[str, Any]]:
    """Return explicit units; legacy memory rows carry no byte measurement."""
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
    """Read traced bytes from extra_info without relabelling pytest timing stats."""
    extra = bench.get("extra_info") or {}
    name = bench.get("fullname", bench["name"]).replace("\\", "/")
    legacy_memory = "test_bench_memory.py" in name or bench["name"] in LEGACY_MEMORY_BENCHMARKS
    result = {
        "median": bench["stats"]["median"],
        "unit": "seconds",
        "workload_version": extra.get("workload_version", 1),
    }
    if extra.get("measurement") == BYTES_MEASUREMENT:
        result.update(median=extra.get(BYTES_MEASUREMENT), unit="bytes")
    elif legacy_memory:
        result.update(median=None, unit="bytes", reason="legacy artifact has no tracemalloc byte measurement")
    value = result["median"]
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
        raise ValueError(f"Invalid {result['unit']} measurement for {bench['name']}: {value!r}")
    return result


def _collect(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    """Map each benchmark to its measurement in every run file that contains it."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        for name, row in _load(path).items():
            runs.setdefault(name, []).append(row)
    return runs


def _medians(rows: list[dict[str, Any]]) -> list[float]:
    return [row["median"] for row in rows]


def _incomparable(master: list[dict[str, Any]], branch: list[dict[str, Any]]) -> str | None:
    """Explain why the runs of a benchmark measured on both sides do not describe the same workload."""
    if not master or not branch:
        return None  # judge() reports a benchmark measured on one side only
    rows = master + branch
    reasons = [row["reason"] for row in rows if row["median"] is None]
    if reasons:
        return str(reasons[0])
    units = sorted({row["unit"] for row in rows})
    if len(units) > 1:
        return f"different units: {' / '.join(units)}"
    if len({row["workload_version"] for row in rows}) > 1:
        return "workload definition changed; a new baseline is required for this benchmark"
    if min(_medians(master)) == 0:
        return "zero baseline: relative change is undefined; absolute test budget still applies"
    return None


def judge(master: Sequence[float], branch: Sequence[float], threshold: float) -> tuple[str, float | None]:
    """Classify one benchmark from its per-run medians on each side (master medians must be positive).

    ``REGRESSED`` needs the lowest branch median above the highest master median by more
    than ``threshold``, ``IMPROVED`` the highest branch median below the lowest master
    median by more than ``threshold``; anything in between is ``UNCHANGED``. A benchmark
    with no run on one side is ``NEW`` or ``REMOVED``. The returned change is the relative
    gap between the nearest edges of the two spreads, or None when nothing moved.
    """
    if not master:
        return NEW, None
    if not branch:
        return REMOVED, None
    slowdown = (min(branch) - max(master)) / max(master)
    if slowdown > threshold and not math.isclose(slowdown, threshold, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE):
        return REGRESSED, slowdown
    speedup = (max(branch) - min(master)) / min(master)
    if speedup < -threshold:
        return IMPROVED, speedup
    return UNCHANGED, None


def _bypass_requested() -> bool:
    """Only the run that adding the bypass label triggers skips the gate; every later run measures again."""
    return os.environ.get("PR_EVENT_ACTION") == "labeled" and os.environ.get("PR_EVENT_LABEL") == BYPASS_LABEL


def _format_delta(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct * 100:.1f}%"


def _spread(rows: list[dict[str, Any]]) -> str:
    low, high = min(_medians(rows)), max(_medians(rows))
    return f"{low:.6g}" if low == high else f"{low:.6g}..{high:.6g}"


def _describe(name: str, master: list[dict[str, Any]], branch: list[dict[str, Any]], change: float | None) -> str:
    if change is None:
        return name
    unit = branch[0]["unit"]
    return f"{name}  median {_spread(master)} -> {_spread(branch)} {unit} ({_format_delta(change)})"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", nargs="+", type=Path, required=True, help="benchmark JSON of each master run")
    parser.add_argument("--branch", nargs="+", type=Path, required=True, help="benchmark JSON of each PR branch run")
    parser.add_argument(
        "--threshold",
        type=float,
        default=REGRESSION_THRESHOLD,
        help=f"Fractional gap beyond the master spread that fails the gate (default: {REGRESSION_THRESHOLD})",
    )
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and non-negative")
    return args


def _report_regressions(lines: list[str]) -> int:
    if _bypass_requested():
        print(
            f"[WARN] Regressions detected; this run was triggered by adding '{BYPASS_LABEL}', bypassing:",
            file=sys.stderr,
        )
        for line in lines:
            print(f"  - {line}", file=sys.stderr)
        return 0
    print("[FAIL] Performance regressions detected:", file=sys.stderr)
    for line in lines:
        print(f"  [REGRESSION] {line}", file=sys.stderr)
    print(
        "\nIf this regression is intentional and accepted:\n"
        "  - Document the trade-off in the PR description\n"
        f"  - Add the '{BYPASS_LABEL}' label: it bypasses only the run that adding it starts,\n"
        "    so every later push is measured again\n",
        file=sys.stderr,
    )
    return 1


def _report(verdicts: dict[str, list[str]], incomparable: int, threshold: float) -> int:
    """Print the verdicts and return the exit code."""
    if verdicts.get(REMOVED):
        print(f"[WARN] Benchmarks present in master but missing from branch: {verdicts[REMOVED]}", file=sys.stderr)
    if verdicts.get(NEW):
        print(f"[INFO] New benchmarks added by branch: {verdicts[NEW]}")
    compared = sum(len(verdicts.get(verdict, [])) for verdict in (REGRESSED, IMPROVED, UNCHANGED))
    print(f"\nBenchmarks compared: {compared} ({incomparable} explicitly incomparable)")
    print(f"Threshold: {threshold * 100:.0f}% beyond the spread of the master runs\n")
    if verdicts.get(IMPROVED):
        print("Improvements (lower latency or memory, no action needed):")
        for line in verdicts[IMPROVED]:
            print(f"  [IMPROVED] {line}")
        print()
    if verdicts.get(REGRESSED):
        return _report_regressions(verdicts[REGRESSED])
    if not compared:
        print("[ERROR] No comparable measurements; cannot establish a regression verdict.", file=sys.stderr)
        return 2
    print("[OK] No comparable benchmarks regressed beyond threshold.")
    return 0


def main() -> int:
    """Return 0 when nothing regressed (or the bypass applies), 1 on a regression, 2 without a verdict."""
    args = _parse_args()
    master, branch = _collect(args.master), _collect(args.branch)
    verdicts: dict[str, list[str]] = {}
    incomparable = 0
    for name in sorted(master.keys() | branch.keys()):
        master_rows, branch_rows = master.get(name, []), branch.get(name, [])
        reason = _incomparable(master_rows, branch_rows)
        if reason:
            print(f"[INCOMPARABLE] {name}: {reason}")
            incomparable += 1
            continue
        verdict, change = judge(_medians(master_rows), _medians(branch_rows), args.threshold)
        verdicts.setdefault(verdict, []).append(_describe(name, master_rows, branch_rows, change))
    return _report(verdicts, incomparable, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
