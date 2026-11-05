"""Backwards-compatibility regression tests (Pillar 6: Versioning).

These tests freeze promises that existing user setups rely on:

1. **Legacy config files** load the way the server loads ``config.yaml`` at
   startup (``_load_yaml_config`` into ``Config``) with every setting they make
   in effect: the ``config.example.yaml`` shipped with v3.6.0 and v3.7.0,
   which catches a changed default it relies on, and a config in the same
   layout with a non-default value for every key, which catches a key that
   stops being read.
2. **Public Python signatures and exception classes** that callers bind to.

The MCP tool contract (tool names, parameters in order, types, required
parameters, defaults) is checked in the quality gate by
scripts/check_mcp_contract.py against the tools that master's code registers.

When a real breaking change is required:

    1. Bump MAJOR version
    2. Document migration path in CHANGELOG
    3. Update or quarantine the affected legacy fixture (with a
       migration helper if appropriate)
    4. Ask a maintainer for the ``skip-api-surface`` label
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import mcp_server.config as config_module

LEGACY_CONFIGS = Path(__file__).parent / "fixtures" / "legacy_configs"
# Provenance: config.example.yaml at 79276b4 (v3.6.0) and 25b69ce (v3.7.0), the
# same blob 3c42f58, unchanged from v3.6.0 to v3.7.0. The repository has no v3.x tags.
SHIPPED_V3_EXAMPLE = "v3.6.0-v3.7.0_config.example.yaml"
# Every setting the shipped example makes, as Config exposes it. The empty
# mappings disable categories, keyword routing and query expansion, as the
# example's comments say.
SHIPPED_V3_SETTINGS = {
    "supported_formats": [
        ".md",
        ".txt",
        ".pdf",
        ".docx",
        ".py",
        ".c",
        ".h",
        ".cpp",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".json",
        ".xml",
    ],
    "exclude_patterns": [],
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_dim": 384,
    "gpu_mode": "false",
    "reranker_enabled": True,
    "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
    "reranker_top_k_multiplier": 3,
    "default_results": 5,
    "max_results": 20,
    "collection_name": "knowledge_base",
    "category_mappings": {},
    "keyword_routes": {},
    "query_expansions": {},
}
SHIPPED_V3_PATHS = {
    "documents_dir": "documents",
    "data_dir": "data",
    "chroma_dir": "data/chroma_db",
    "models_cache_dir": "models_cache",
}
# Hand-written in that example's layout (provenance in the file's header). Every
# value differs from today's default, so a key that stops being read fails;
# the shipped example above catches a changed default instead.
V3_LAYOUT_FIXTURE = "v3.7.0_layout_non_default_values.yaml"
V3_LAYOUT_SETTINGS = {
    "supported_formats": [".md", ".py"],
    "exclude_patterns": ["node_modules", ".git", "__pycache__", ".venv", "*.tmp", "drafts/*"],
    "chunk_size": 1500,
    "chunk_overlap": 300,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_dim": 768,
    "gpu_mode": "true",
    "reranker_enabled": False,
    "reranker_model": "BAAI/bge-reranker-base",
    "reranker_top_k_multiplier": 5,
    "default_results": 8,
    "max_results": 50,
    "collection_name": "work_kb",
    "category_mappings": {"recipes/italian": "italian", "recipes": "cooking"},
    "keyword_routes": {"cooking": ["pasta", "risotto"]},
    "query_expansions": {"js": ["javascript"]},
}
V3_LAYOUT_PATHS = {
    "documents_dir": "notes",
    "data_dir": "index",
    "chroma_dir": "index/chroma_db",
    "models_cache_dir": "cache",
}
LEGACY_CASES = [
    pytest.param((SHIPPED_V3_EXAMPLE, SHIPPED_V3_SETTINGS, SHIPPED_V3_PATHS), id="shipped-v3-example"),
    pytest.param((V3_LAYOUT_FIXTURE, V3_LAYOUT_SETTINGS, V3_LAYOUT_PATHS), id="v3-layout-non-default-values"),
]


def load_legacy_config(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> config_module.Config:
    """Load a fixture as ``config.yaml`` the way the server does at startup, isolated in ``tmp_path``."""
    (tmp_path / "config.yaml").write_bytes((LEGACY_CONFIGS / name).read_bytes())
    monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config_module, "_yaml", config_module._load_yaml_config())
    return config_module.Config()


# ---------------------------------------------------------------------------
# Layer 1: legacy YAML configs load through Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", LEGACY_CASES)
def test_legacy_config_loads_with_every_setting_in_effect(case, tmp_path, monkeypatch):
    """Paths, documents, models, search and mappings keep the values the legacy config sets."""
    fixture, settings, paths = case
    loaded = load_legacy_config(fixture, tmp_path, monkeypatch)
    expected = {**settings, **{name: tmp_path / path for name, path in paths.items()}}
    assert {name: getattr(loaded, name) for name in expected} == expected


# ---------------------------------------------------------------------------
# Layer 2: public Python signatures
# ---------------------------------------------------------------------------


def test_folder_parameters_are_optional_and_preserve_existing_positional_call():
    """An additive folder scope must not change existing callers' arguments."""
    from mcp_server import server

    signature = inspect.signature(server.search_knowledge)
    bound = signature.bind("query", 5, None, 0.3, 0.0, True, "auto")
    bound.apply_defaults()
    assert bound.arguments["search_method"] == "auto"
    assert bound.arguments["include_folders"] is None
    assert bound.arguments["exclude_folders"] is None


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
