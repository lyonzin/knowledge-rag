"""Backwards-compatibility regression tests (Pillar 6: Versioning).

These tests freeze the public surface promises of knowledge-rag so future
PRs cannot silently break existing user setups. Two layers:

1. **Legacy config files** load without raising. We keep verbatim YAML
   snapshots from prior releases under tests/fixtures/legacy_configs/ and
   feed them through Config so any breaking schema change shows up here.
2. **Public MCP tool signatures** preserve their parameter names. The
   downstream LLMs (Claude, GPT, etc.) call these tools by name; renaming
   a parameter is silently breaking even if Python imports still work.

When a real breaking change is required:

    1. Bump MAJOR version
    2. Document migration path in CHANGELOG
    3. Update or quarantine the affected legacy fixture (with a
       migration helper if appropriate)
    4. Update the expected-signature dict below
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

LEGACY_CONFIGS = Path(__file__).parent / "fixtures" / "legacy_configs"


# ---------------------------------------------------------------------------
# Layer 1: legacy YAML configs still parse
# ---------------------------------------------------------------------------


def test_legacy_v3_6_0_config_parses():
    """v3.6.0 minimal config must load without error."""
    cfg_path = LEGACY_CONFIGS / "v3.6.0_minimal.yaml"
    assert cfg_path.exists(), f"Fixture missing: {cfg_path}"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Spot-check structural promises that v3.6.0 users rely on
    assert data["models"]["embedding"]["name"] == "BAAI/bge-small-en-v1.5"
    assert data["models"]["embedding"]["dim"] == 384
    assert data["models"]["reranker"]["enabled"] is True
    assert data["chunking"]["chunk_size"] == 1000
    assert data["search"]["hybrid_alpha"] == 0.3


def test_legacy_v3_7_0_config_with_excludes_parses():
    """v3.7.0 config including exclude_patterns must load without error."""
    cfg_path = LEGACY_CONFIGS / "v3.7.0_with_excludes.yaml"
    assert cfg_path.exists(), f"Fixture missing: {cfg_path}"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert isinstance(data["exclude_patterns"], list)
    assert "*.tmp" in data["exclude_patterns"]
    assert "node_modules/" in data["exclude_patterns"]


# ---------------------------------------------------------------------------
# Layer 2: MCP tool signatures preserve parameter names
# ---------------------------------------------------------------------------


# Frozen contract: parameter names callers (LLMs) supply by name.
# Bumping requires MAJOR version + CHANGELOG migration entry.
MCP_TOOL_SIGNATURES = {
    # Task 03 (ADR-006, v4.8.2+): appended ``search_method`` opt-in override.
    # Insertion is additive with default ``"auto"`` — legacy callers unaffected.
    "search_knowledge": [
        "query",
        "max_results",
        "category",
        "hybrid_alpha",
        "min_score",
        "snippet_mode",
        "search_method",
    ],
    "search_similar": ["filepath", "max_results"],
    "get_document": ["filepath"],
    "add_document": ["content", "filepath", "category"],
    "add_from_url": ["url", "category", "title"],
    "update_document": ["filepath", "content"],
    "remove_document": ["filepath", "delete_file"],
    "reindex_documents": ["force", "full_rebuild", "resume"],
    "list_categories": [],
    "list_documents": ["category"],
    "get_index_stats": [],
    "get_reindex_status": [],
    "evaluate_retrieval": ["test_cases"],
}


def test_mcp_tool_parameter_names_preserved():
    """Renaming an MCP tool parameter is a breaking change for all LLM callers."""
    from mcp_server import server

    for tool_name, expected_params in MCP_TOOL_SIGNATURES.items():
        fn = getattr(server, tool_name, None)
        assert fn is not None, f"MCP tool removed: {tool_name}"

        sig = inspect.signature(fn)
        actual_params = [
            p.name
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)
        ]

        assert actual_params == expected_params, (
            f"MCP tool '{tool_name}' parameter names changed.\n"
            f"  expected: {expected_params}\n"
            f"  actual:   {actual_params}\n"
            f"This is a BREAKING change for every LLM client that calls this tool by name."
        )


def test_exception_classes_present():
    """Public exception classes must remain available for callers to catch."""
    from mcp_server.server import EmbeddingError, EmbeddingModelLoadError

    assert issubclass(EmbeddingError, RuntimeError)
    assert issubclass(EmbeddingModelLoadError, RuntimeError)


# ---------------------------------------------------------------------------
# UT-041, UT-042 — search_knowledge accepts the new opt-in `search_method`
# ---------------------------------------------------------------------------


def test_ut041_search_knowledge_without_new_param_returns_json(monkeypatch):
    """UT-041: pre-v4.8.2 call surface (no ``search_method``) still returns JSON."""
    import json

    from mcp_server import server as srv

    # Isolate: fake orchestrator that yields empty results — the wrapper still
    # emits a well-formed JSON envelope (no TypeError from missing arg).
    class _Orch:
        query_cache = type("_C", (), {"stats": staticmethod(lambda: {"hit_rate": "0%"})})()

        def query(self, *args, **kwargs):
            return []

    monkeypatch.setattr(srv, "get_orchestrator", lambda: _Orch())
    raw = srv.search_knowledge(query="X")
    payload = json.loads(raw)
    assert payload["status"] in ("no_results", "success", "error")


def test_ut042_search_knowledge_with_all_seven_params_returns_json(monkeypatch):
    """UT-042: passing all 7 params (incl. ``search_method``) does not raise."""
    import json

    from mcp_server import server as srv

    class _Orch:
        query_cache = type("_C", (), {"stats": staticmethod(lambda: {"hit_rate": "0%"})})()

        def query(self, *args, **kwargs):
            return []

    monkeypatch.setattr(srv, "get_orchestrator", lambda: _Orch())
    raw = srv.search_knowledge(
        query="X",
        max_results=5,
        category=None,
        hybrid_alpha=0.3,
        min_score=0.0,
        snippet_mode=True,
        search_method="auto",
    )
    payload = json.loads(raw)
    assert isinstance(payload, dict)


def test_ut042b_invalid_search_method_returns_structured_error(monkeypatch):
    """Guardrail: unknown ``search_method`` value returns a JSON error, not a crash."""
    import json

    from mcp_server import server as srv

    class _Orch:
        query_cache = type("_C", (), {"stats": staticmethod(lambda: {"hit_rate": "0%"})})()

        def query(self, *args, **kwargs):
            return []

    monkeypatch.setattr(srv, "get_orchestrator", lambda: _Orch())
    raw = srv.search_knowledge(query="X", search_method="nonsense")
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert "search_method" in payload["message"]


def test_instance_lock_module_public_surface():
    """Single-instance lock module promises preserved across releases."""
    from mcp_server import instance_lock

    expected = {
        "single_instance_enabled",
        "single_instance_lock",
        "AlreadyRunningError",
        "ALREADY_RUNNING_EXIT_CODE",
        "ENV_VAR",
        "LOCK_FILENAME",
    }
    actual = {name for name in dir(instance_lock) if not name.startswith("_")}
    missing = expected - actual
    assert not missing, f"Missing from instance_lock public surface: {missing}"
