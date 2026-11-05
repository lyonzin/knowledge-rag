"""Compare interleaved benchmark runs of master and the PR branch; fail on a consistent >10% regression.

Pillar 5 — Scalability: the perf-bench job runs ``bench/`` four times, in the
order branch, master, master, branch, and writes one pytest-benchmark JSON per
run. Each run gives every benchmark one median: wall time, or tracemalloc peak
bytes for the memory benchmarks. Sub-100 µs medians move by ±15–19% between
runs of identical code, so one median per side cannot tell noise from a change.

A benchmark regresses only when its lowest branch median is above its highest
master median by more than ``REGRESSION_THRESHOLD`` (10% by default); a byte
measurement must also grow by more than ``BYTES_FLOOR``. Spreads that overlap
are noise and pass; improvements are reported, never blocking. A benchmark that
master has and the branch lacks, that some runs of a side lack, or whose runs
cannot be compared fails too: otherwise renaming, skipping or re-versioning a
slow benchmark would pass.

Exit codes: 0 the gate passes (or the bypass applies), 1 the gate fails,
2 the input is unusable (missing or empty file, duplicate benchmark name).

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
from typing import Any, NoReturn

REGRESSION_THRESHOLD = 0.10  # 10%
# tracemalloc gives the same bytes on every run, so 10% of a 23 KB peak (2.4 KB) is a real change
# but too small to block a PR; a byte regression must also grow by more than this.
BYTES_FLOOR = 32 * 1024
BYPASS_LABEL = "skip-perf-gate"
BYTES_MEASUREMENT = "tracemalloc_peak_bytes"
LEGACY_RSS_MEASUREMENT = "rss_peak_delta_bytes"
# test_bench_orchestrator_idle_rss is the pre-tracemalloc name that legacy artifacts still carry.
MEMORY_BENCHMARKS = {
    "test_bench_orchestrator_idle_rss",
    "test_bench_orchestrator_idle_memory",
    "test_bench_query_cache_5000_entries",
}
# An exact 10% change computes as 0.10000000000000009; it must not count as above the threshold.
FLOAT_TOLERANCE = 1e-12
REGRESSED, IMPROVED, UNCHANGED, NEW, REMOVED = "regressed", "improved", "unchanged", "new", "removed"
INCOMPARABLE, INCOMPLETE, WIDE_SPREAD = "incomparable", "incomplete", "wide spread"
COMPARED = (REGRESSED, IMPROVED, UNCHANGED)
FAILURE_TAGS = (
    (REGRESSED, "REGRESSION"),
    (REMOVED, "REMOVED"),
    (INCOMPLETE, "INCOMPLETE"),
    (INCOMPARABLE, "INCOMPARABLE"),
)
BYPASS_HELP = f"""
If the change is intentional and accepted:
  - Document the trade-off in the PR description.
  - Add the '{BYPASS_LABEL}' label. It bypasses only the run that adding it starts, so it must be
    the last label added: after any other label change, remove it and add it again. Every later
    push is measured again.
"""


def _input_error(message: str) -> NoReturn:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(2)


def _load(path: Path) -> dict[str, dict[str, Any]]:
    """Return one measurement per benchmark name; a duplicate name would hide one of its rows."""
    if not path.exists():
        _input_error(f"Benchmark file missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:  # JSONDecodeError or UnicodeDecodeError: empty, truncated or not text
        _input_error(f"Benchmark file is not valid JSON: {path} ({error})")
    out: dict[str, dict[str, Any]] = {}
    for bench in payload.get("benchmarks", []):
        if bench["name"] in out:
            _input_error(f"Duplicate benchmark name {bench['name']!r} in {path}")
        out[bench["name"]] = _measurement(bench)
    if not out:
        _input_error(f"No benchmarks found in {path}")
    return out


def _measurement(bench: dict[str, Any]) -> dict[str, Any]:
    """Read traced bytes from extra_info without relabelling pytest timing stats."""
    extra = bench.get("extra_info") or {}
    name = bench.get("fullname", bench["name"]).replace("\\", "/")
    memory = (
        "test_bench_memory.py" in name
        or bench["name"] in MEMORY_BENCHMARKS
        or extra.get("measurement") == LEGACY_RSS_MEASUREMENT
    )
    result = {
        "median": bench["stats"]["median"],
        "unit": "seconds",
        "workload_version": extra.get("workload_version", 1),
    }
    if extra.get("measurement") == BYTES_MEASUREMENT:
        result.update(median=extra.get(BYTES_MEASUREMENT), unit="bytes")
    elif memory:
        result.update(median=None, unit="bytes", reason="legacy artifact has no tracemalloc byte measurement")
    value = result["median"]
    if value is None:
        result.setdefault("reason", f"no {result['unit']} value recorded")
    elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {result['unit']} measurement for {bench['name']}: {value!r}")
    return result


def _collect(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    """Map each benchmark to one row per run; a run that lacks it holds a ``missing_from`` placeholder."""
    runs = {str(path): _load(path) for path in paths}
    names = {name for rows in runs.values() for name in rows}
    return {name: [rows.get(name, {"missing_from": run}) for run, rows in runs.items()] for name in names}


def _medians(rows: list[dict[str, Any]]) -> list[float]:
    return [row["median"] for row in rows]


def _incomparable(master: list[dict[str, Any]], branch: list[dict[str, Any]]) -> str | None:
    """Explain why a benchmark cannot be judged: a row without its value, or sides measuring different things."""
    rows = master + branch
    reasons = [row["reason"] for row in rows if row["median"] is None]
    if reasons:
        return str(reasons[0])
    if not master or not branch:
        return None  # judge() reports a benchmark measured on one side only
    units = sorted({row["unit"] for row in rows})
    if len(units) > 1:
        return f"different units: {' / '.join(units)}"
    if len({row["workload_version"] for row in rows}) > 1:
        return "workload definition changed; a new baseline is required for this benchmark"
    if min(_medians(master)) == 0:
        return "zero baseline: relative change is undefined; absolute test budget still applies"
    return None


def judge(
    master: Sequence[float], branch: Sequence[float], threshold: float, min_gap: float = 0.0
) -> tuple[str, float | None]:
    """Classify one benchmark from its per-run medians on each side (master medians must be positive).

    ``REGRESSED`` needs the lowest branch median above the highest master median by more
    than ``threshold`` and by more than ``min_gap`` in absolute terms; ``IMPROVED`` needs
    the highest branch median below the lowest master median by more than ``threshold``;
    anything else is ``UNCHANGED``. A benchmark with no run on one side is ``NEW`` or
    ``REMOVED``. The returned change is the relative gap between the nearest edges of
    the two spreads, or None when nothing moved.
    """
    if not master:
        return NEW, None
    if not branch:
        return REMOVED, None
    gap = min(branch) - max(master)
    slowdown = gap / max(master)
    beyond = slowdown > threshold and not math.isclose(
        slowdown, threshold, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE
    )
    if beyond and gap > min_gap:
        return REGRESSED, slowdown
    speedup = (max(branch) - min(master)) / min(master)
    if speedup < -threshold:
        return IMPROVED, speedup
    return UNCHANGED, None


def _bypass_requested() -> bool:
    """Only the run that adding the bypass label triggers skips the gate; every later run measures again."""
    return os.environ.get("PR_EVENT_ACTION") == "labeled" and os.environ.get("PR_EVENT_LABEL") == BYPASS_LABEL


def _value(value: float, unit: str) -> str:
    return f"{value:.0f}" if unit == "bytes" else f"{value:.6g}"


def _spread(rows: list[dict[str, Any]]) -> str:
    unit = rows[0]["unit"]
    low, high = min(_medians(rows)), max(_medians(rows))
    return _value(low, unit) if low == high else f"{_value(low, unit)}..{_value(high, unit)}"


def _describe(name: str, master: list[dict[str, Any]], branch: list[dict[str, Any]], change: float | None) -> str:
    if not branch:
        return f"{name}: measured on master, missing from every branch run"
    if change is None:
        return name
    return f"{name}  median {_spread(master)} -> {_spread(branch)} {branch[0]['unit']} ({change:+.1%})"


def _spread_warning(master: list[dict[str, Any]], threshold: float, min_gap: float) -> str | None:
    """Say how far the branch must move to fail when master's own runs spread wider than the threshold."""
    medians, unit = _medians(master), master[0]["unit"]
    fastest, slowest = min(medians), max(medians)
    if slowest <= fastest * (1 + threshold):
        return None
    fails_above = max(slowest * (1 + threshold), slowest + min_gap)
    return (
        f"master runs spread {slowest / fastest - 1:.1%} ({_spread(master)} {unit}); only a branch above "
        f"{_value(fails_above, unit)} {unit} ({fails_above / fastest - 1:+.1%} over the fastest master run) fails"
    )


def _evaluate(
    name: str, master: list[dict[str, Any]], branch: list[dict[str, Any]], threshold: float
) -> dict[str, str]:
    """Return this benchmark's report lines keyed by verdict, plus a WIDE_SPREAD warning when it applies."""
    missing = [row["missing_from"] for row in master + branch if "missing_from" in row]
    if missing:
        return {INCOMPLETE: f"{name}: missing from {', '.join(missing)}"}
    reason = _incomparable(master, branch)
    if reason:
        return {INCOMPARABLE: f"{name}: {reason}"}
    min_gap = BYTES_FLOOR if (master or branch)[0]["unit"] == "bytes" else 0.0
    verdict, change = judge(_medians(master), _medians(branch), threshold, min_gap)
    lines = {verdict: _describe(name, master, branch, change)}
    warning = _spread_warning(master, threshold, min_gap) if verdict in COMPARED else None
    if warning:
        lines[WIDE_SPREAD] = f"{name}: {warning}"
    return lines


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


def _report_failures(failures: list[tuple[str, str]]) -> int:
    sys.stdout.flush()  # keep the summary above the failures in CI logs
    bypass = _bypass_requested()
    if bypass:
        print(f"[WARN] Gate failures bypassed: this run was triggered by adding '{BYPASS_LABEL}':", file=sys.stderr)
    else:
        print("[FAIL] Performance gate failed:", file=sys.stderr)
    for tag, line in failures:
        print(f"  [{tag}] {line}", file=sys.stderr)
    if bypass:
        return 0
    print(BYPASS_HELP, file=sys.stderr)
    return 1


def _report(report: dict[str, list[str]], threshold: float) -> int:
    """Print every verdict; REGRESSED, REMOVED, INCOMPLETE and INCOMPARABLE rows fail unless bypassed."""
    if report.get(NEW):
        print(f"[INFO] New benchmarks added by branch: {report[NEW]}")
    compared = sum(len(report.get(verdict, [])) for verdict in COMPARED)
    print(f"\nBenchmarks compared: {compared}")
    print(f"Threshold: {threshold:.0%} beyond the spread of the master runs (and {BYTES_FLOOR} bytes for memory)\n")
    for line in report.get(WIDE_SPREAD, []):
        print(f"[WARN] {line}")
    if report.get(IMPROVED):
        print("Improvements (lower latency or memory, no action needed):")
        for line in report[IMPROVED]:
            print(f"  [IMPROVED] {line}")
        print()
    failures = [(tag, line) for verdict, tag in FAILURE_TAGS for line in report.get(verdict, [])]
    if failures:
        return _report_failures(failures)
    print("[OK] No benchmark regressed, went missing or became incomparable.")
    return 0


def main() -> int:
    """Return 0 when the gate passes or the bypass applies, 1 when it fails, 2 for unusable input."""
    args = _parse_args()
    master, branch = _collect(args.master), _collect(args.branch)
    report: dict[str, list[str]] = {}
    for name in sorted(master.keys() | branch.keys()):
        lines = _evaluate(name, master.get(name, []), branch.get(name, []), args.threshold)
        for verdict, line in lines.items():
            report.setdefault(verdict, []).append(line)
    return _report(report, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
