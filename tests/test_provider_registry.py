"""
╭─╴ PROVIDER REGISTRY TESTS ╶────────────────────────────────────╮
│                                                                │
│   Tests for the Fase 2 / A2.2 provider registry: register /    │
│   get / list per ABC, default auto-registration, entry_points  │
│   discovery, orchestrator wiring and clear-error behaviour.    │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.providers import (
    EmbeddingProvider,
    Reranker,
    VectorStore,
    get_embedding,
    get_embedding_class,
    get_reranker,
    get_reranker_class,
    get_vector_store,
    get_vector_store_class,
    is_embedding_registered,
    is_reranker_registered,
    is_vector_store_registered,
    list_embeddings,
    list_rerankers,
    list_vector_stores,
    load_third_party,
    register_embedding,
    register_reranker,
    register_vector_store,
)
from mcp_server.providers import registry as registry_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def preserve_registry():
    """Snapshot / restore the global provider registry.

    Every test that registers a new provider does so under a unique name and
    this fixture wipes the additions in ``finally`` so no test leaks into the
    next. The bundled defaults (``fastembed`` / ``chromadb`` / ``cross_encoder``)
    are preserved because they were registered at import time before the test
    session started.
    """
    emb_snapshot = dict(registry_module._EMBEDDINGS)
    vs_snapshot = dict(registry_module._VECTOR_STORES)
    rr_snapshot = dict(registry_module._RERANKERS)
    tp_flag = registry_module._third_party_loaded

    yield

    registry_module._EMBEDDINGS.clear()
    registry_module._EMBEDDINGS.update(emb_snapshot)
    registry_module._VECTOR_STORES.clear()
    registry_module._VECTOR_STORES.update(vs_snapshot)
    registry_module._RERANKERS.clear()
    registry_module._RERANKERS.update(rr_snapshot)
    registry_module._third_party_loaded = tp_flag


# ---------------------------------------------------------------------------
# Fake providers — minimal implementations for registration tests
# ---------------------------------------------------------------------------


class _FakeEmbedding:
    """Bare-bones ``EmbeddingProvider`` for registry tests."""

    name = "fake-embedding"
    dimension = 8

    def __init__(self, prefix: str = "fake") -> None:
        self.prefix = prefix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * self.dimension


class _FakeVectorStore:
    """Bare-bones ``VectorStore`` for registry tests."""

    name = "fake-vector-store"

    def __init__(self, *, storage: Optional[Dict[str, Any]] = None) -> None:
        self.storage = storage if storage is not None else {}

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        for i, id_ in enumerate(ids):
            self.storage[id_] = {
                "embedding": embeddings[i] if embeddings else None,
                "metadata": metadatas[i] if metadatas else None,
                "document": documents[i] if documents else None,
            }

    def query(self, query_embeddings, n_results, where=None):
        return {"ids": [list(self.storage.keys())[:n_results]]}

    def get(self, ids=None, include=None):
        return {"ids": ids or list(self.storage.keys())}

    def delete(self, ids):
        for id_ in ids:
            self.storage.pop(id_, None)

    def count(self) -> int:
        return len(self.storage)


class _FakeReranker:
    """Bare-bones ``Reranker`` for registry tests."""

    name = "fake-reranker"

    def rerank(self, query, documents, top_k):
        return list(documents)[:top_k]


# ---------------------------------------------------------------------------
# Bundled defaults — regression guard
# ---------------------------------------------------------------------------


class TestBundledDefaults:
    """The historical stack must remain auto-registered on import."""

    def test_fastembed_registered_by_default(self):
        assert is_embedding_registered("fastembed"), "FastEmbed should be registered on import"
        assert "fastembed" in list_embeddings()

    def test_chromadb_registered_by_default(self):
        assert is_vector_store_registered("chromadb"), "ChromaDB should be registered on import"
        assert "chromadb" in list_vector_stores()

    def test_cross_encoder_registered_by_default(self):
        assert is_reranker_registered("cross_encoder"), "cross_encoder should be registered on import"
        assert "cross_encoder" in list_rerankers()

    def test_fastembed_class_resolves_to_fastembedembeddings(self):
        from mcp_server.retrieval.embeddings import FastEmbedEmbeddings

        assert get_embedding_class("fastembed") is FastEmbedEmbeddings

    def test_chromadb_class_resolves_to_chromavectorstore(self):
        from mcp_server.storage.chroma import ChromaVectorStore

        assert get_vector_store_class("chromadb") is ChromaVectorStore

    def test_cross_encoder_class_resolves_to_crossencoderreranker(self):
        from mcp_server.retrieval.rerank import CrossEncoderReranker

        assert get_reranker_class("cross_encoder") is CrossEncoderReranker


# ---------------------------------------------------------------------------
# Register / get / list per ABC
# ---------------------------------------------------------------------------


class TestEmbeddingRegistration:
    """Register embeddings via decorator, direct call, and factory."""

    def test_direct_registration(self, preserve_registry):
        register_embedding("test-embed-direct", _FakeEmbedding)
        assert is_embedding_registered("test-embed-direct")
        assert "test-embed-direct" in list_embeddings()

    def test_decorator_registration(self, preserve_registry):
        @register_embedding("test-embed-deco")
        class _MyEmbed(_FakeEmbedding):
            pass

        assert is_embedding_registered("test-embed-deco")
        assert get_embedding_class("test-embed-deco") is _MyEmbed

    def test_factory_returns_instance(self, preserve_registry):
        register_embedding("test-embed-fac", _FakeEmbedding)
        instance = get_embedding("test-embed-fac", prefix="hello")
        assert isinstance(instance, _FakeEmbedding)
        assert instance.prefix == "hello"

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError) as excinfo:
            get_embedding("this-does-not-exist")
        assert "this-does-not-exist" in str(excinfo.value)
        assert "Registered" in str(excinfo.value)

    def test_empty_name_rejected(self, preserve_registry):
        with pytest.raises(ValueError):
            register_embedding("", _FakeEmbedding)

    def test_list_returns_sorted(self, preserve_registry):
        register_embedding("zzz-embed", _FakeEmbedding)
        register_embedding("aaa-embed", _FakeEmbedding)
        names = list_embeddings()
        idx_aaa = names.index("aaa-embed")
        idx_zzz = names.index("zzz-embed")
        assert idx_aaa < idx_zzz


class TestVectorStoreRegistration:
    """Register vector stores via decorator, direct call, and factory."""

    def test_direct_registration(self, preserve_registry):
        register_vector_store("test-vs-direct", _FakeVectorStore)
        assert is_vector_store_registered("test-vs-direct")

    def test_decorator_registration(self, preserve_registry):
        @register_vector_store("test-vs-deco")
        class _MyStore(_FakeVectorStore):
            pass

        assert get_vector_store_class("test-vs-deco") is _MyStore

    def test_factory_returns_instance(self, preserve_registry):
        register_vector_store("test-vs-fac", _FakeVectorStore)
        instance = get_vector_store("test-vs-fac")
        assert isinstance(instance, _FakeVectorStore)
        assert instance.count() == 0

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_vector_store("bogus-vs")


class TestRerankerRegistration:
    """Register rerankers via decorator, direct call, and factory."""

    def test_direct_registration(self, preserve_registry):
        register_reranker("test-rr-direct", _FakeReranker)
        assert is_reranker_registered("test-rr-direct")

    def test_decorator_registration(self, preserve_registry):
        @register_reranker("test-rr-deco")
        class _MyRR(_FakeReranker):
            pass

        assert get_reranker_class("test-rr-deco") is _MyRR

    def test_factory_returns_instance(self, preserve_registry):
        register_reranker("test-rr-fac", _FakeReranker)
        instance = get_reranker("test-rr-fac")
        assert isinstance(instance, _FakeReranker)

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_reranker("bogus-rr")


# ---------------------------------------------------------------------------
# Protocol conformance sanity checks
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """runtime_checkable Protocols only verify attribute presence."""

    def test_fake_embedding_conforms_to_protocol(self):
        instance = _FakeEmbedding()
        assert isinstance(instance, EmbeddingProvider)

    def test_fake_vector_store_conforms_to_protocol(self):
        instance = _FakeVectorStore()
        assert isinstance(instance, VectorStore)

    def test_fake_reranker_conforms_to_protocol(self):
        instance = _FakeReranker()
        assert isinstance(instance, Reranker)


# ---------------------------------------------------------------------------
# Entry-point discovery (mocked)
# ---------------------------------------------------------------------------


class TestThirdPartyDiscovery:
    """``load_third_party`` walks importlib.metadata entry_points groups."""

    def test_load_third_party_iterates_all_three_groups(self, preserve_registry, monkeypatch):
        calls: List[str] = []

        def fake_entry_points(group: str):
            calls.append(group)
            return []

        # Force re-discovery so the guard doesn't short-circuit.
        registry_module._third_party_loaded = False
        monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)

        load_third_party()

        assert set(calls) == {
            "knowledge_rag.embeddings",
            "knowledge_rag.vector_stores",
            "knowledge_rag.rerankers",
        }

    def test_load_third_party_invokes_plugin_load(self, preserve_registry, monkeypatch):
        """Each entry point should be loaded exactly once."""
        plugin_module = MagicMock()
        plugin_module.__name__ = "plugin_class_not_callable"

        # Make it look like a *class* so the registrar path is skipped after load().
        class _PluginClass:  # pragma: no cover - loaded but not called
            pass

        ep = MagicMock()
        ep.name = "my-plugin"
        ep.load = MagicMock(return_value=_PluginClass)

        def fake_entry_points(group: str):
            if group == "knowledge_rag.embeddings":
                return [ep]
            return []

        registry_module._third_party_loaded = False
        monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)

        load_third_party()

        ep.load.assert_called_once()

    def test_load_third_party_calls_callable_plugin(self, preserve_registry, monkeypatch):
        """When the entry point resolves to a callable (not a class), call it."""
        register_called: List[bool] = []

        def _plugin_register():
            register_called.append(True)
            register_embedding("plugin-registered", _FakeEmbedding)

        ep = MagicMock()
        ep.name = "callable-plugin"
        ep.load = MagicMock(return_value=_plugin_register)

        def fake_entry_points(group: str):
            if group == "knowledge_rag.embeddings":
                return [ep]
            return []

        registry_module._third_party_loaded = False
        monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)

        load_third_party()

        assert register_called == [True]
        assert is_embedding_registered("plugin-registered")

    def test_load_third_party_isolates_plugin_failures(self, preserve_registry, monkeypatch, caplog):
        """A broken plugin must not stop the next one from loading."""
        good_registered: List[bool] = []

        def _good_plugin():
            good_registered.append(True)

        bad_ep = MagicMock()
        bad_ep.name = "broken"
        bad_ep.load = MagicMock(side_effect=RuntimeError("boom"))

        good_ep = MagicMock()
        good_ep.name = "good"
        good_ep.load = MagicMock(return_value=_good_plugin)

        def fake_entry_points(group: str):
            if group == "knowledge_rag.embeddings":
                return [bad_ep, good_ep]
            return []

        registry_module._third_party_loaded = False
        monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)

        with caplog.at_level("WARNING", logger="mcp_server.providers.registry"):
            load_third_party()

        assert good_registered == [True], "good plugin must still run after bad one fails"

    def test_load_third_party_is_idempotent_by_default(self, preserve_registry, monkeypatch):
        """Repeated calls without force=True must not re-enumerate."""
        call_count: List[int] = [0]

        def fake_entry_points(group: str):
            call_count[0] += 1
            return []

        registry_module._third_party_loaded = False
        monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)

        load_third_party()
        first = call_count[0]
        load_third_party()  # should short-circuit
        assert call_count[0] == first, "second call must not re-enumerate entry_points"

        load_third_party(force=True)  # explicit re-run allowed
        assert call_count[0] > first


# ---------------------------------------------------------------------------
# Orchestrator wiring — defaults still hit the historical code path
# ---------------------------------------------------------------------------


class TestOrchestratorDefaults:
    """``KnowledgeOrchestrator`` must instantiate the historical stack when
    ``embedding_provider`` / ``vector_store`` / ``reranker_provider`` keep
    their default values."""

    def test_default_config_uses_fastembed_chromadb_cross_encoder(self):
        from mcp_server.config import config

        assert config.embedding_provider == "fastembed"
        assert config.vector_store == "chromadb"
        assert config.reranker_provider == "cross_encoder"

    def test_orchestrator_uses_late_bind_fastembed_when_default(self, tmp_path, monkeypatch):
        """When embedding_provider='fastembed', the orchestrator must go
        through :func:`_resolve_embeddings_class` so tests can patch
        ``mcp_server.server.FastEmbedEmbeddings``. This is the exact contract
        the ``mock_embedding`` fixture depends on."""

        from mcp_server import config as cfg
        from mcp_server.retrieval.orchestrator import _resolve_embeddings_class

        docs = tmp_path / "documents"
        docs.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        chroma = data / "chroma_db"
        chroma.mkdir()
        models = tmp_path / "models_cache"
        models.mkdir()

        monkeypatch.setattr(cfg.config, "documents_dir", docs)
        monkeypatch.setattr(cfg.config, "data_dir", data)
        monkeypatch.setattr(cfg.config, "chroma_dir", chroma)
        monkeypatch.setattr(cfg.config, "models_cache_dir", models)
        monkeypatch.setattr(cfg.config, "transport", "stdio")
        monkeypatch.setattr(cfg.config, "embedding_provider", "fastembed")
        monkeypatch.setattr(cfg.config, "vector_store", "chromadb")
        monkeypatch.setattr(cfg.config, "reranker_provider", "cross_encoder")

        # Sentinel embedding class installed via the same patch target the
        # historical mock_embedding fixture uses.
        sentinel = MagicMock(name="SentinelEmbeddings")
        sentinel_instance = MagicMock()
        sentinel_instance.name.return_value = "sentinel"
        sentinel_instance._dim = 384
        sentinel_instance.__call__ = MagicMock(return_value=[[0.0] * 384])
        sentinel_instance.embed_documents = MagicMock(return_value=[[0.0] * 384])
        sentinel_instance.embed_query = MagicMock(return_value=[[0.0] * 384])
        sentinel.return_value = sentinel_instance

        # ChromaDB Collection has a strict embedding_function contract.
        # Use the same _FakeEmbeddings shape as test_dedup.py.
        class _FakeEmbeddingsChroma:
            _dim = 384
            is_legacy = False

            def __call__(self, input):
                return [[0.1] * 384 for _ in input]

            @staticmethod
            def name():
                return "fake"

            @staticmethod
            def build_from_config(config):  # noqa: ARG004
                return _FakeEmbeddingsChroma()

            @staticmethod
            def get_config():
                return {}

            @staticmethod
            def validate_config_update(old, new):  # noqa: ARG004
                pass

        with patch("mcp_server.server.FastEmbedEmbeddings", _FakeEmbeddingsChroma):
            resolved = _resolve_embeddings_class()
            assert resolved is _FakeEmbeddingsChroma, (
                "The late-bind hook must observe the patch — this is what the mock_embedding fixture relies on."
            )

            # A2.2: use the refactored orchestrator in mcp_server.retrieval —
            # the monolithic ``mcp_server.server.KnowledgeOrchestrator`` predates
            # the provider registry and is scheduled for removal in the follow-up
            # A2.3 cleanup.
            from mcp_server.retrieval.orchestrator import KnowledgeOrchestrator

            orch = KnowledgeOrchestrator()
            assert isinstance(orch.embed_fn, _FakeEmbeddingsChroma)
            assert orch.collection is not None  # ChromaDB collection alive
            assert orch.chroma_client is not None

    def test_orchestrator_rejects_unknown_embedding_provider(self, tmp_path, monkeypatch):
        """Setting an unregistered provider name must raise clearly."""
        from mcp_server import config as cfg

        docs = tmp_path / "documents"
        docs.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        chroma = data / "chroma_db"
        chroma.mkdir()
        models = tmp_path / "models_cache"
        models.mkdir()

        monkeypatch.setattr(cfg.config, "documents_dir", docs)
        monkeypatch.setattr(cfg.config, "data_dir", data)
        monkeypatch.setattr(cfg.config, "chroma_dir", chroma)
        monkeypatch.setattr(cfg.config, "models_cache_dir", models)
        monkeypatch.setattr(cfg.config, "transport", "stdio")
        monkeypatch.setattr(cfg.config, "embedding_provider", "nonexistent-provider-xyz")

        # A2.2: use the refactored orchestrator — the monolith in
        # ``mcp_server.server`` predates the provider registry and doesn't
        # dispatch through it.
        from mcp_server.retrieval.orchestrator import KnowledgeOrchestrator

        with pytest.raises(ValueError, match="nonexistent-provider-xyz"):
            KnowledgeOrchestrator()


# ---------------------------------------------------------------------------
# ChromaVectorStore Protocol delegation
# ---------------------------------------------------------------------------


class TestChromaVectorStoreWrapper:
    """The bundled ``ChromaVectorStore`` wrapper must implement the Protocol
    and expose the raw ChromaDB handles."""

    def _make_store(self, tmp_path):
        from mcp_server.storage.chroma import ChromaVectorStore

        class _FakeEmbeddingsChroma:
            _dim = 384
            is_legacy = False

            def __call__(self, input):
                return [[0.1] * 384 for _ in input]

            @staticmethod
            def name():
                return "wrap-test"

            @staticmethod
            def build_from_config(config):  # noqa: ARG004
                return _FakeEmbeddingsChroma()

            @staticmethod
            def get_config():
                return {}

            @staticmethod
            def validate_config_update(old, new):  # noqa: ARG004
                pass

        return ChromaVectorStore(
            path=tmp_path / "chroma",
            collection_name="wrap_test",
            embedding_function=_FakeEmbeddingsChroma(),
            enable_wal=False,
        )

    def test_wrapper_exposes_client_and_collection(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.client is not None
        assert store.collection is not None
        assert store.name == "chromadb"

    def test_wrapper_count_delegates(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.count() == 0

    def test_wrapper_add_get_delete_roundtrip(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add(
            ids=["a", "b"],
            documents=["hello world one", "hello world two"],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
        )
        assert store.count() == 2

        fetched = store.get(ids=["a"], include=["documents"])
        assert "a" in fetched["ids"]

        store.delete(ids=["a"])
        assert store.count() == 1
