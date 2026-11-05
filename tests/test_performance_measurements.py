"""Regression gates and dashboard must never confuse traced bytes with seconds."""

import json
import sys

import pytest

from bench.dashboard.build import build_payload
from bench.test_bench_memory import _trace_allocations
from scripts import check_perf_regression
from scripts.audit_requirements import dependency_pins
from scripts.check_perf_regression import IMPROVED, NEW, REGRESSED, REMOVED, UNCHANGED, judge

MIB = 1024 * 1024
THRESHOLD = check_perf_regression.REGRESSION_THRESHOLD


def _row(name, seconds, traced=None, version=1):
    row = {"name": name, "stats": {"median": seconds}}
    if traced is not None:
        row["extra_info"] = {
            "measurement": "tracemalloc_peak_bytes",
            "tracemalloc_peak_bytes": traced,
            "tracemalloc_retained_bytes": 1024,
            "memory_budget_bytes": 80 * MIB,
            "workload_version": version,
        }
    return row


def _gate(tmp_path, monkeypatch, master_runs, branch_runs):
    """Run the CLI with one JSON file per run; each run is a list of benchmark rows."""
    argv = ["check_perf_regression.py"]
    for side, runs in (("master", master_runs), ("branch", branch_runs)):
        argv.append(f"--{side}")
        for index, rows in enumerate(runs, start=1):
            path = tmp_path / f"{side}-{index}.json"
            path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")
            argv.append(str(path))
    monkeypatch.setattr(sys, "argv", argv)
    return check_perf_regression.main()


@pytest.fixture(autouse=True)
def _outside_a_labeled_run(monkeypatch):
    """Gate tests start as an ordinary push unless they set the triggering event themselves."""
    for name in ("PR_EVENT_ACTION", "PR_EVENT_LABEL", "PR_LABELS"):
        monkeypatch.delenv(name, raising=False)


def test_given_overlapping_spreads_when_judged_then_noise_passes():
    # #263 failed on fts5_lexical_cold[CWE-79] at +15.1% (33.9 -> 39.0 us) from a single pair of runs.
    assert judge([33.9, 39.5], [39.0, 34.4], THRESHOLD) == (UNCHANGED, None)


def test_given_every_branch_run_beyond_the_master_spread_when_judged_then_it_regresses():
    verdict, change = judge([1.0, 1.05], [1.2, 1.25], THRESHOLD)
    assert (verdict, change) == (REGRESSED, pytest.approx(1.2 / 1.05 - 1))


def test_given_every_branch_run_below_the_master_spread_when_judged_then_it_improves():
    assert judge([1.0, 1.05], [0.8, 0.85], THRESHOLD)[0] == IMPROVED


@pytest.mark.parametrize(("master", "branch", "verdict"), [([], [1.0, 1.1], NEW), ([1.0, 1.1], [], REMOVED)])
def test_given_a_benchmark_missing_on_one_side_when_judged_then_it_is_not_compared(master, branch, verdict):
    assert judge(master, branch, THRESHOLD) == (verdict, None)


def test_given_noise_one_run_pair_would_fail_when_gated_interleaved_then_it_passes(tmp_path, monkeypatch):
    master = [[_row("search", 1.0)], [_row("search", 1.2)]]
    branch = [[_row("search", 1.15)], [_row("search", 1.25)]]
    assert _gate(tmp_path, monkeypatch, master, branch) == 0


def test_given_a_benchmark_missing_on_one_side_when_gated_then_the_rest_still_decides(tmp_path, monkeypatch, capsys):
    master = [[_row("search", 1), _row("removed", 1)]]
    branch = [[_row("search", 1.2), _row("added", 1)]]
    assert _gate(tmp_path, monkeypatch, master, branch) == 1
    output = capsys.readouterr()
    assert "['added']" in output.out
    assert "['removed']" in output.err


def test_given_a_missing_master_file_when_gated_then_it_fails(tmp_path, monkeypatch):
    branch = tmp_path / "branch-1.json"
    branch.write_text(json.dumps({"benchmarks": [_row("search", 1)]}), encoding="utf-8")
    argv = ["check_perf_regression.py", "--master", str(tmp_path / "master-1.json"), "--branch", str(branch)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as raised:
        check_perf_regression.main()
    assert raised.value.code == 2


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (("labeled", "skip-perf-gate"), 0),
        (("synchronize", ""), 1),
        (("opened", ""), 1),
        (("unlabeled", "skip-perf-gate"), 1),
        (("labeled", "skip-changelog"), 1),
    ],
)
def test_given_the_bypass_label_on_the_pr_then_only_the_run_adding_it_bypasses(tmp_path, monkeypatch, event, expected):
    monkeypatch.setenv("PR_LABELS", "skip-perf-gate")
    monkeypatch.setenv("PR_EVENT_ACTION", event[0])
    monkeypatch.setenv("PR_EVENT_LABEL", event[1])
    assert _gate(tmp_path, monkeypatch, [[_row("search", 1)]], [[_row("search", 2)]]) == expected


def test_given_four_times_the_objects_when_traced_then_the_peak_grows_fourfold():
    # Workload version 2 sampled RSS, which read one 4 KiB page whatever the workload allocated:
    # small objects reuse pages the allocator already holds.
    small, large = (
        _trace_allocations(lambda count=count: [bytearray(100) for _ in range(count)])["peak_bytes"]
        for count in (10_000, 40_000)
    )
    assert large / small == pytest.approx(4, rel=0.05)


def test_legacy_memory_row_is_incomparable_but_latency_still_blocks(tmp_path, monkeypatch, capsys):
    old = [_row("test_bench_query_cache_5000_entries", 1), _row("search", 1)]
    new = [_row("test_bench_query_cache_5000_entries", 1, traced=9000, version=3), _row("search", 1.2)]
    assert _gate(tmp_path, monkeypatch, [old], [new]) == 1
    output = capsys.readouterr()
    assert "legacy artifact has no tracemalloc byte measurement" in output.out
    assert "search" in output.err
    assert "+20.0%" in output.err


def test_traced_growth_blocks_using_bytes_even_if_workload_time_improves(tmp_path, monkeypatch, capsys):
    old = [_row("memory", 9, traced=10_000, version=3)]
    new = [_row("memory", 1, traced=12_000, version=3)]
    assert _gate(tmp_path, monkeypatch, [old], [new]) == 1
    assert "10000 -> 12000 bytes" in capsys.readouterr().err


def test_memory_timer_is_not_a_byte_metric(tmp_path, monkeypatch):
    old = [_row("memory", 1, traced=10_000, version=3)]
    new = [_row("memory", 9, traced=10_000, version=3)]
    assert _gate(tmp_path, monkeypatch, [old], [new]) == 0


def test_exactly_ten_percent_latency_growth_is_not_above_threshold(tmp_path, monkeypatch):
    assert _gate(tmp_path, monkeypatch, [[_row("search", 1)]], [[_row("search", 1.1)]]) == 0


def test_new_workload_skips_only_its_row(tmp_path, monkeypatch, capsys):
    old = [_row("memory", 1, traced=1000, version=2), _row("search", 1)]
    new = [_row("memory", 1, traced=9000, version=3), _row("search", 1.2)]
    assert _gate(tmp_path, monkeypatch, [old], [new]) == 1
    assert "workload definition changed" in capsys.readouterr().out


def test_no_comparable_measurements_cannot_claim_success(tmp_path, monkeypatch, capsys):
    old = [_row("memory", 1, traced=0, version=3)]
    new = [_row("memory", 1, traced=0, version=3)]
    assert _gate(tmp_path, monkeypatch, [old], [new]) == 2
    assert "zero baseline" in capsys.readouterr().out


@pytest.mark.parametrize("value", [float("nan"), -1, "50 MB"])
def test_invalid_byte_measurements_fail_validation(value):
    with pytest.raises(ValueError, match="Invalid bytes"):
        check_perf_regression._measurement(_row("memory", 1, traced=value))


def test_dashboard_keeps_byte_units_and_marks_legacy_unavailable():
    rows = [
        _row("search", 0.001),
        _row("memory", 9, traced=10_000, version=3),
        _row("test_bench_orchestrator_idle_rss", 9),
    ]
    result = build_payload({"benchmarks": rows}, "abc123", "3.9.0")
    by_name = {row["name"]: row for row in result["results"]}
    assert by_name["memory"]["measurement_unit"] == "bytes"
    assert by_name["memory"]["measurement_value"] == 10_000
    assert by_name["memory"]["budget_bytes"] == 80 * MIB
    assert by_name["memory"]["median_ns"] is None
    assert by_name["memory"]["ops"] is None
    assert by_name["search"]["measurement_value"] == 1_000_000
    assert by_name["search"]["measurement_unit"] == "nanoseconds"
    assert by_name["test_bench_orchestrator_idle_rss"]["measurement_value"] is None


def test_audit_exports_requested_and_transitive_dependencies_only():
    report = {
        "version": "1",
        "install": [
            {"metadata": {"name": "knowledge_rag", "version": "99.0.0"}, "requested": True},
            {"metadata": {"name": "Direct.Pkg", "version": "2.0"}, "requested": True},
            {"metadata": {"name": "transitive-pkg", "version": "3.1"}, "requested": False},
        ],
    }
    assert dependency_pins(report) == ["direct-pkg==2.0", "transitive-pkg==3.1"]


@pytest.mark.parametrize("report", [{}, {"version": "1", "install": []}])
def test_audit_does_not_accept_an_empty_dependency_graph(report):
    with pytest.raises(ValueError):
        dependency_pins(report)


def test_audit_rejects_conflicting_resolved_versions():
    report = {
        "version": "1",
        "install": [
            {"metadata": {"name": "some-pkg", "version": "1.0"}},
            {"metadata": {"name": "some_pkg", "version": "2.0"}},
        ],
    }
    with pytest.raises(ValueError, match="Conflicting resolved versions"):
        dependency_pins(report)
