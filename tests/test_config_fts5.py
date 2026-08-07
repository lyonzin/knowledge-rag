"""Config UT-030..038: FTS5 fast-path YAML nested fields (Fase 1 / Task 01).

Each test spins up an isolated ``Config`` instance from a synthetic YAML doc
via ``monkeypatch`` on the module-level ``_yaml`` dict, so we exercise the
real ``default_factory`` + ``__post_init__`` validators without touching the
process-wide ``config`` singleton.
"""

from __future__ import annotations

import re

import pytest

from mcp_server import config as config_module


def _reload_config(monkeypatch, yaml_payload: dict):
    """Replace module-level ``_yaml`` and return a fresh Config instance."""
    monkeypatch.setattr(config_module, "_yaml", yaml_payload)
    return config_module.Config()


# ---------------------------------------------------------------------------
# UT-030..038
# ---------------------------------------------------------------------------


def test_ut030_defaults_when_section_missing(monkeypatch):
    """UT-030: YAML without search.lexical_fast_path → all defaults kick in."""
    cfg = _reload_config(monkeypatch, {})
    assert cfg.fts5_enabled is False
    assert cfg.fts5_min_hits == 3
    assert cfg.fts5_rerank_enabled is False
    assert cfg.fts5_patterns == [
        r"[A-Z]{2,}-\d+",
        r"CVE-\d{4}-\d+",
        r"^[a-f0-9]{32,64}$",
    ]


def test_ut031_defaults_when_section_empty(monkeypatch):
    """UT-031: search.lexical_fast_path: {} — defaults still apply."""
    cfg = _reload_config(monkeypatch, {"search": {"lexical_fast_path": {}}})
    assert cfg.fts5_enabled is False
    assert cfg.fts5_min_hits == 3
    assert cfg.fts5_rerank_enabled is False
    assert len(cfg.fts5_patterns) == 3


def test_ut032_invalid_regex_pattern_raises(monkeypatch):
    """UT-032: invalid regex in patterns fails startup with re.error."""
    yaml_payload = {"search": {"lexical_fast_path": {"enabled": True, "patterns": ["("]}}}
    with pytest.raises(re.error):
        _reload_config(monkeypatch, yaml_payload)


def test_ut033_string_enabled_falls_back_to_default(monkeypatch, capsys):
    """UT-033: enabled='true' (string) is rejected; default False wins + WARN log."""
    cfg = _reload_config(
        monkeypatch,
        {"search": {"lexical_fast_path": {"enabled": "true"}}},
    )
    assert cfg.fts5_enabled is False
    captured = capsys.readouterr().out
    assert "wrong type" in captured or "invalid" in captured


def test_ut034_empty_patterns_list_warns(monkeypatch, capsys):
    """UT-034: patterns: [] is valid but logs a WARN that no query will classify."""
    cfg = _reload_config(
        monkeypatch,
        {"search": {"lexical_fast_path": {"enabled": True, "patterns": []}}},
    )
    assert cfg.fts5_patterns == []
    captured = capsys.readouterr().out
    assert "empty" in captured.lower() or "will never" in captured.lower()


def test_ut035_custom_pattern_appended(monkeypatch):
    """UT-035: custom PROJ-\\d{3,5} pattern lands intact in fts5_patterns."""
    yaml_payload = {
        "search": {
            "lexical_fast_path": {
                "patterns": [
                    r"[A-Z]{2,}-\d+",
                    r"CVE-\d{4}-\d+",
                    r"^[a-f0-9]{32,64}$",
                    r"PROJ-\d{3,5}",
                ]
            }
        }
    }
    cfg = _reload_config(monkeypatch, yaml_payload)
    assert r"PROJ-\d{3,5}" in cfg.fts5_patterns


def test_ut036_malformed_lookbehind_raises(monkeypatch):
    """UT-036: pattern '(?<=' triggers re.error at startup."""
    yaml_payload = {"search": {"lexical_fast_path": {"patterns": ["(?<="]}}}
    with pytest.raises(re.error):
        _reload_config(monkeypatch, yaml_payload)


def test_ut037_overly_broad_pattern_warns(monkeypatch, capsys):
    """UT-037: pattern '.+' logs a broad-pattern warning."""
    yaml_payload = {"search": {"lexical_fast_path": {"patterns": [".+"]}}}
    _reload_config(monkeypatch, yaml_payload)
    captured = capsys.readouterr().out
    assert "overly broad" in captured or "broad" in captured.lower()


def test_ut038_high_pattern_count_warns(monkeypatch, capsys):
    """UT-038: >20 patterns logs a router-performance warning."""
    patterns = [rf"CODE{i}-\d+" for i in range(25)]
    yaml_payload = {"search": {"lexical_fast_path": {"patterns": patterns}}}
    _reload_config(monkeypatch, yaml_payload)
    captured = capsys.readouterr().out
    assert "high pattern count" in captured.lower() or "25" in captured
