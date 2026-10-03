"""Regression gates and dashboard must never confuse RSS bytes with seconds."""

import json
import sys

import pytest

from bench.dashboard.build import build_payload
from scripts import check_perf_regression
from scripts.audit_requirements import dependency_pins


def _row(name, seconds, rss=None, version=1):
    row = {"name": name, "stats": {"median": seconds}}
    if rss is not None:
        row["extra_info"] = {
            "measurement": "rss_peak_delta_bytes",
            "rss_peak_delta_bytes": rss,
            "rss_retained_delta_bytes": 1024,
            "rss_budget_bytes": 80 * 1024 * 1024,
            "workload_version": version,
        }
    return row


def _gate(tmp_path, monkeypatch, master, branch):
    paths = [tmp_path / "master.json", tmp_path / "branch.json"]
    for path, rows in zip(paths, [master, branch]):
        path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")
    monkeypatch.delenv("PR_LABELS", raising=False)
    monkeypatch.setattr(sys, "argv", ["check_perf_regression.py", *map(str, paths)])
    return check_perf_regression.main()


def test_legacy_rss_is_incomparable_but_latency_still_blocks(tmp_path, monkeypatch, capsys):
    old = [_row("test_bench_query_cache_5000_entries", 1), _row("search", 1)]
    new = [_row("test_bench_query_cache_5000_entries", 1, rss=9000, version=2), _row("search", 1.2)]
    assert _gate(tmp_path, monkeypatch, old, new) == 1
    output = capsys.readouterr()
    assert "legacy artifact has no RSS byte measurement" in output.out
    assert "search" in output.err
    assert "+20.0%" in output.err


def test_rss_growth_blocks_using_bytes_even_if_workload_time_improves(tmp_path, monkeypatch, capsys):
    old = [_row("memory", 9, rss=10_000, version=2)]
    new = [_row("memory", 1, rss=12_000, version=2)]
    assert _gate(tmp_path, monkeypatch, old, new) == 1
    assert "10000 -> 12000 bytes" in capsys.readouterr().err


def test_memory_timer_is_not_an_rss_metric(tmp_path, monkeypatch):
    old = [_row("memory", 1, rss=10_000, version=2)]
    new = [_row("memory", 9, rss=10_000, version=2)]
    assert _gate(tmp_path, monkeypatch, old, new) == 0


def test_exactly_ten_percent_latency_growth_is_not_above_threshold(tmp_path, monkeypatch):
    assert _gate(tmp_path, monkeypatch, [_row("search", 1)], [_row("search", 1.1)]) == 0


def test_new_workload_skips_only_its_row(tmp_path, monkeypatch, capsys):
    old = [_row("memory", 1, rss=1000, version=1), _row("search", 1)]
    new = [_row("memory", 1, rss=9000, version=2), _row("search", 1.2)]
    assert _gate(tmp_path, monkeypatch, old, new) == 1
    assert "workload definition changed" in capsys.readouterr().out


def test_no_comparable_measurements_cannot_claim_success(tmp_path, monkeypatch, capsys):
    old = [_row("memory", 1, rss=0, version=2)]
    new = [_row("memory", 1, rss=0, version=2)]
    assert _gate(tmp_path, monkeypatch, old, new) == 2
    assert "zero baseline" in capsys.readouterr().out


@pytest.mark.parametrize("value", [float("nan"), -1, "50 MB"])
def test_invalid_byte_measurements_fail_validation(value):
    with pytest.raises(ValueError, match="Invalid bytes"):
        check_perf_regression._measurement(_row("memory", 1, rss=value))


def test_dashboard_keeps_rss_units_and_marks_legacy_unavailable():
    rows = [
        _row("search", 0.001),
        _row("memory", 9, rss=10_000, version=2),
        _row("test_bench_orchestrator_idle_rss", 9),
    ]
    result = build_payload({"benchmarks": rows}, "abc123", "3.9.0")
    by_name = {row["name"]: row for row in result["results"]}
    assert by_name["memory"]["measurement_unit"] == "bytes"
    assert by_name["memory"]["measurement_value"] == 10_000
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
