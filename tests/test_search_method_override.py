"""IT-010..IT-014 — per-request ``search_method`` override on ``search_knowledge``.

ADR-006 pins the semantics of every override value:
- ``"auto"`` (default): router decides.
- ``"hybrid"``: skip router; force hybrid pipeline (kill switch).
- ``"fts5"``: skip router; force FTS5. Errors out if feature disabled or
  index not ready — never silently falls back (that would defeat the
  debug/testing purpose of the override).

Tests exercise the MCP tool wrapper end-to-end (JSON envelope) so the
Fts5NotReadyError path shows up exactly as the client would see it.
"""

from __future__ import annotations

import json

import pytest


def _stub_orch_from_test_search(monkeypatch, fts5, router, *, fake_docs=None, fake_metadatas=None):
    """Reuse the light dispatch orchestrator helper from tests/test_search.py."""
    from tests.test_search import _build_dispatch_orch

    return _build_dispatch_orch(
        monkeypatch,
        fts5_index=fts5,
        query_router=router,
        fake_docs=fake_docs or {},
        fake_metadatas=fake_metadatas or {},
    )


def _install_orch_singleton(monkeypatch, orch):
    """Bind ``get_orchestrator()`` to a specific orchestrator instance.

    The MCP tool wrapper looks up ``get_orchestrator`` in the server module
    namespace on every call, so replacing the callable is enough — no need
    to touch the private singleton lock.
    """
    import mcp_server.server as srv

    monkeypatch.setattr(srv, "get_orchestrator", lambda: orch)


def _sample_docs():
    return {f"chunk_{i}": f"doc {i} mentioning CVE-2021-4034" for i in range(5)}, {
        f"chunk_{i}": {
            "source": f"/{i}",
            "filename": f"{i}.md",
            "category": "security",
            "chunk_index": 0,
            "keywords": "",
        }
        for i in range(5)
    }


@pytest.fixture
def fts5_and_router():
    from tests.test_search import _FakeFts5, _FakeRouter

    hits = [(f"chunk_{i}", 10 - i) for i in range(5)]
    return _FakeFts5(hits=hits, ready=True), _FakeRouter("lexical")


def test_it010_hybrid_override_skips_router(monkeypatch, fts5_and_router):
    """IT-010: ``search_method="hybrid"`` skips FTS5 even for lexical queries."""
    from mcp_server import server as srv

    fts5, router = fts5_and_router
    docs, metas = _sample_docs()
    orch = _stub_orch_from_test_search(monkeypatch, fts5, router, fake_docs=docs, fake_metadatas=metas)
    _install_orch_singleton(monkeypatch, orch)

    raw = srv.search_knowledge(query="CVE-2021-4034", search_method="hybrid")
    payload = json.loads(raw)

    # Router never consulted for explicit hybrid override.
    assert router.classify_calls == []
    # FTS5 must not have been queried either.
    assert fts5.search_calls == []
    # Empty hybrid pipeline → no_results envelope.
    assert payload.get("status") in ("no_results", "success")


def test_it011_hybrid_override_when_feature_off_is_noop(monkeypatch, fts5_and_router):
    """IT-011: enabled=False + hybrid override = same behavior as default."""
    from mcp_server import server as srv

    fts5, router = fts5_and_router
    docs, metas = _sample_docs()
    orch = _stub_orch_from_test_search(monkeypatch, fts5, router, fake_docs=docs, fake_metadatas=metas)
    monkeypatch.setattr(srv.config, "fts5_enabled", False)
    _install_orch_singleton(monkeypatch, orch)

    raw = srv.search_knowledge(query="CVE-2021-4034", search_method="hybrid")
    payload = json.loads(raw)

    assert fts5.search_calls == []
    assert payload.get("status") in ("no_results", "success")


def test_it012_fts5_override_uses_fast_path(monkeypatch, fts5_and_router):
    """IT-012: enabled + fts5 ready + explicit ``fts5`` → FTS5 dispatched."""
    from mcp_server import server as srv

    fts5, router = fts5_and_router
    docs, metas = _sample_docs()
    orch = _stub_orch_from_test_search(monkeypatch, fts5, router, fake_docs=docs, fake_metadatas=metas)
    _install_orch_singleton(monkeypatch, orch)

    raw = srv.search_knowledge(query="X", search_method="fts5")
    payload = json.loads(raw)

    assert fts5.search_calls, "FTS5 must have been queried for explicit override"
    # Router bypassed on explicit search_method.
    assert router.classify_calls == []
    assert payload.get("status") == "success"
    assert payload["results"][0]["search_method"] == "fts5"


def test_it013_fts5_override_on_prose_still_dispatches(monkeypatch, fts5_and_router):
    """IT-013: enabled + prose query + explicit fts5 → FTS5 runs (quality is the user's problem)."""
    from mcp_server import server as srv

    fts5, router = fts5_and_router
    docs, metas = _sample_docs()
    orch = _stub_orch_from_test_search(monkeypatch, fts5, router, fake_docs=docs, fake_metadatas=metas)
    _install_orch_singleton(monkeypatch, orch)

    raw = srv.search_knowledge(query="how does OAuth work", search_method="fts5")
    payload = json.loads(raw)

    assert fts5.search_calls, "explicit fts5 override must dispatch even on prose"
    assert payload.get("status") == "success"


def test_it014_fts5_override_when_disabled_returns_error_json(monkeypatch, fts5_and_router):
    """IT-014: enabled=False + explicit fts5 → JSON error, no silent fallback."""
    from mcp_server import server as srv

    fts5, router = fts5_and_router
    orch = _stub_orch_from_test_search(monkeypatch, fts5, router)
    monkeypatch.setattr(srv.config, "fts5_enabled", False)
    _install_orch_singleton(monkeypatch, orch)

    raw = srv.search_knowledge(query="X", search_method="fts5")
    payload = json.loads(raw)

    assert payload["status"] == "error"
    assert "disabled" in payload["error"]
    assert "lexical_fast_path" in payload["error"]
    # Suggestion is always attached so the user has an obvious recovery step.
    assert "auto" in payload.get("suggestion", "")
    # FTS5 was NEVER queried — the wrapper short-circuited on config.
    assert fts5.search_calls == []
