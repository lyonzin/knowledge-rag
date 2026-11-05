"""Regression gates and dashboard must never confuse traced bytes with seconds."""

import json
import sys
import tracemalloc

import pytest

from bench.dashboard.build import build_payload
from bench.test_bench_memory import _benchmark_memory, _trace_peak
from scripts import check_perf_regression
from scripts.audit_requirements import dependency_pins
from scripts.check_perf_regression import BYTES_FLOOR, IMPROVED, NEW, REGRESSED, REMOVED, UNCHANGED, judge

MIB = 1024 * 1024
THRESHOLD = check_perf_regression.REGRESSION_THRESHOLD
MISSING_VALUE = {"measurement": "tracemalloc_peak_bytes", "workload_version": 3}


def _row(name, seconds, traced=None, version=1):
    row = {"name": name, "stats": {"median": seconds}}
    if traced is not None:
        row["extra_info"] = {
            "measurement": "tracemalloc_peak_bytes",
            "tracemalloc_peak_bytes": traced,
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


class _RecordingBenchmark:
    """Just enough of pytest-benchmark's fixture to run the memory benchmark helper."""

    def __init__(self):
        self.extra_info = {}

    def pedantic(self, target, iterations, rounds):
        for _ in range(iterations * rounds):
            target()


@pytest.fixture(autouse=True)
def _outside_a_labeled_run(monkeypatch):
    """Gate tests start as an ordinary push unless they set the triggering event themselves."""
    for name in ("PR_EVENT_ACTION", "PR_EVENT_LABEL"):
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


@pytest.mark.parametrize(
    ("master", "branch", "verdict"),
    [
        (23_752, 28_502, UNCHANGED),  # +20%, but only 4.6 KiB more
        (838_808, 1_006_570, REGRESSED),  # +20% and 164 KiB more
    ],
)
def test_given_traced_bytes_when_judged_then_a_regression_must_also_exceed_the_floor(master, branch, verdict):
    assert judge([master, master], [branch, branch], THRESHOLD, BYTES_FLOOR)[0] == verdict


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ((_row("memory", 1, traced=23_752, version=3), _row("memory", 1, traced=28_502, version=3)), 0),
        ((_row("memory", 1, traced=838_808, version=3), _row("memory", 1, traced=1_006_570, version=3)), 1),
        ((_row("search", 1e-6), _row("search", 1.2e-6)), 1),
    ],
)
def test_given_a_twenty_percent_growth_when_gated_then_only_bytes_need_the_floor(tmp_path, monkeypatch, case, expected):
    assert _gate(tmp_path, monkeypatch, [[case[0]]], [[case[1]]]) == expected


def test_given_noise_one_run_pair_would_fail_when_gated_interleaved_then_it_passes(tmp_path, monkeypatch):
    master = [[_row("search", 1.0)], [_row("search", 1.2)]]
    branch = [[_row("search", 1.15)], [_row("search", 1.25)]]
    assert _gate(tmp_path, monkeypatch, master, branch) == 0


def test_given_master_runs_spread_beyond_the_threshold_when_gated_then_it_warns_of_the_floor(
    tmp_path, monkeypatch, capsys
):
    master = [[_row("search", 1.0)], [_row("search", 1.5)]]
    branch = [[_row("search", 1.2)], [_row("search", 1.3)]]
    assert _gate(tmp_path, monkeypatch, master, branch) == 0
    assert "only a branch above 1.65 seconds (+65.0% over the fastest master run) fails" in capsys.readouterr().out


def test_given_a_benchmark_missing_from_the_branch_when_gated_then_it_fails(tmp_path, monkeypatch, capsys):
    master = [[_row("search", 1), _row("renamed", 1)]]
    branch = [[_row("search", 1), _row("renamed_v2", 1)]]
    assert _gate(tmp_path, monkeypatch, master, branch) == 1
    assert "[REMOVED] renamed: measured on master, missing from every branch run" in capsys.readouterr().err


def test_given_a_new_benchmark_when_gated_then_it_is_only_reported(tmp_path, monkeypatch, capsys):
    assert _gate(tmp_path, monkeypatch, [[_row("search", 1)]], [[_row("search", 1), _row("added", 1)]]) == 0
    assert "['added']" in capsys.readouterr().out


@pytest.mark.parametrize(
    "case",
    [
        (
            _row("test_bench_query_cache_5000_entries", 1),
            _row("test_bench_query_cache_5000_entries", 1, traced=9000, version=3),
            "legacy artifact has no tracemalloc byte measurement",
        ),
        (_row("memory", 1, traced=1000, version=2), _row("memory", 1, traced=9000, version=3), "workload definition"),
        (_row("memory", 1, traced=0, version=3), _row("memory", 1, traced=0, version=3), "zero baseline"),
        ({"name": "memory", "stats": {"median": 1}, "extra_info": MISSING_VALUE}, _row("memory", 1), "no bytes value"),
        (_row("memory", 1), _row("memory", 1, traced=9000, version=1), "different units: bytes / seconds"),
    ],
)
def test_given_runs_that_cannot_be_compared_when_gated_then_it_fails_with_the_reason(
    tmp_path, monkeypatch, capsys, case
):
    master, branch, reason = case
    assert _gate(tmp_path, monkeypatch, [[master]], [[branch]]) == 1
    assert f"[INCOMPARABLE] {master['name']}: {reason}" in capsys.readouterr().err


def test_given_a_duplicate_benchmark_name_in_one_run_when_gated_then_the_input_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as raised:
        _gate(tmp_path, monkeypatch, [[_row("search", 1), _row("search", 2)]], [[_row("search", 1)]])
    assert raised.value.code == 2


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
def test_given_gate_failures_then_only_the_run_adding_the_bypass_label_passes(tmp_path, monkeypatch, event, expected):
    # The label stays on the PR in every case; only the event that started the run differs.
    monkeypatch.setenv("PR_LABELS", "skip-perf-gate")
    monkeypatch.setenv("PR_EVENT_ACTION", event[0])
    monkeypatch.setenv("PR_EVENT_LABEL", event[1])
    master = [[_row("search", 1), _row("removed", 1), _row("memory", 1, traced=1000, version=2)]]
    branch = [[_row("search", 2), _row("memory", 1, traced=1000, version=3)]]
    assert _gate(tmp_path, monkeypatch, master, branch) == expected


def test_given_a_failure_when_reported_then_it_says_the_bypass_label_must_be_added_last(tmp_path, monkeypatch, capsys):
    _gate(tmp_path, monkeypatch, [[_row("search", 1)]], [[_row("search", 2)]])
    assert "the last label added" in capsys.readouterr().err


def test_given_four_times_the_objects_when_traced_then_the_peak_grows_fourfold():
    # Workload version 2 sampled RSS, which read one 4 KiB page whatever the workload allocated:
    # small objects reuse pages the allocator already holds.
    def peak(count):
        # The least of three runs, so another thread allocating during one of them cannot inflate it.
        return min(_trace_peak(lambda: [bytearray(100) for _ in range(count)]) for _ in range(3))

    assert peak(40_000) / peak(10_000) == pytest.approx(4, rel=0.05)


def test_given_tracemalloc_already_tracing_when_traced_then_it_refuses_and_keeps_the_outer_trace():
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="already tracing"):
            _trace_peak(list)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_given_the_memory_helper_output_then_gate_and_dashboard_read_the_same_bytes():
    benchmark = _RecordingBenchmark()
    _benchmark_memory(benchmark, lambda: bytearray(MIB), 2 * MIB)
    row = {
        "name": "test_bench_contract",
        "fullname": "bench/test_bench_memory.py::test_bench_contract",
        "stats": {"median": 0.001},
        "extra_info": benchmark.extra_info,
    }
    gate = check_perf_regression._measurement(row)
    dashboard = build_payload({"benchmarks": [row]}, "abc123", "9.9.9")["results"][0]
    read = (gate["unit"], gate["median"], dashboard["measurement_value"], dashboard["budget_bytes"])
    assert read == ("bytes", pytest.approx(MIB, rel=0.01), pytest.approx(MIB, rel=0.01), 2 * MIB)


def test_traced_growth_blocks_using_bytes_even_if_workload_time_improves(tmp_path, monkeypatch, capsys):
    old = [_row("memory", 9, traced=1_000_000, version=3)]
    new = [_row("memory", 1, traced=1_200_000, version=3)]
    assert _gate(tmp_path, monkeypatch, [old], [new]) == 1
    assert "1000000 -> 1200000 bytes (+20.0%)" in capsys.readouterr().err


def test_memory_timer_is_not_a_byte_metric(tmp_path, monkeypatch):
    old = [_row("memory", 1, traced=10_000, version=3)]
    new = [_row("memory", 9, traced=10_000, version=3)]
    assert _gate(tmp_path, monkeypatch, [old], [new]) == 0


def test_exactly_ten_percent_latency_growth_is_not_above_threshold(tmp_path, monkeypatch):
    assert _gate(tmp_path, monkeypatch, [[_row("search", 1)]], [[_row("search", 1.1)]]) == 0


@pytest.mark.parametrize("value", [float("nan"), -1, "50 MB"])
def test_invalid_byte_measurements_fail_validation(value):
    with pytest.raises(ValueError, match="Invalid bytes"):
        check_perf_regression._measurement(_row("memory", 1, traced=value))


def test_dashboard_keeps_byte_units_and_marks_legacy_unavailable():
    rows = [
        _row("search", 0.001),
        _row("memory", 9, traced=10_000, version=3),
        _row("test_bench_orchestrator_idle_memory", 9),
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
    assert by_name["test_bench_orchestrator_idle_memory"]["measurement_value"] is None


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
