"""GH #287: configurable HNSW ef_search + self-recall probe in get_index_stats.

Coverage mirrors the mitigation design:
- ``search.ef_search`` flows from config.yaml into the ``configuration`` kwarg
  of every collection creation path (production, staging, destructive rebuild).
- ``self_recall`` is computed from Chroma's own answers (mocked collection to
  test the fraction math, real Chroma to test the happy path), cached for
  ``_SELF_RECALL_CACHE_TTL_SECONDS`` and dropped together with the query cache
  on every index mutation.
"""

import json
import uuid
from unittest.mock import MagicMock

import pytest

# ----------------------------------------------------------------- config


def test_ef_search_default_is_100(monkeypatch):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {})
    assert config_module.Config().ef_search == 100


def test_ef_search_from_yaml_is_respected(monkeypatch):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {"search": {"ef_search": 400}})
    assert config_module.Config().ef_search == 400


@pytest.mark.parametrize("raw", [0, -3, 1.5, "abc", True])
def test_given_invalid_ef_search_when_loading_then_it_falls_back_to_100(raw, monkeypatch):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {"search": {"ef_search": raw}})
    assert config_module.Config().ef_search == 100


def test_given_ef_search_zero_when_loading_then_a_warn_is_printed(monkeypatch, capsys):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {"search": {"ef_search": 0}})
    assert config_module.Config().ef_search == 100

    assert "[WARN] ef_search=0 invalid, using 100" in capsys.readouterr().err


# -------------------------------------------------- collection creation


def _bare_orchestrator():
    from mcp_server import server

    orch = object.__new__(server.KnowledgeOrchestrator)
    orch.embed_fn = MagicMock()
    orch.chroma_client = MagicMock()
    return orch


def test_given_ef_search_config_when_opening_collection_then_hnsw_configuration_is_passed(monkeypatch):
    from mcp_server import server

    orch = _bare_orchestrator()
    monkeypatch.setattr(server.config, "ef_search", 400)

    orch._open_collection()

    kwargs = orch.chroma_client.get_or_create_collection.call_args.kwargs
    assert kwargs["configuration"] == {"hnsw": {"ef_search": 400}}


def test_given_ef_search_config_when_creating_staging_then_hnsw_configuration_is_passed(monkeypatch):
    from mcp_server import server

    orch = _bare_orchestrator()
    monkeypatch.setattr(server.config, "ef_search", 250)

    orch._create_staging_collection(12345)

    kwargs = orch.chroma_client.get_or_create_collection.call_args.kwargs
    assert kwargs["configuration"] == {"hnsw": {"ef_search": 250}}
    assert kwargs["name"].endswith("__staging_12345")


def test_given_ef_search_400_when_collection_is_created_then_chroma_stores_it(monkeypatch):
    import chromadb

    from mcp_server import server

    class _Ef:
        def __call__(self, input):
            return [[0.0, 0.0] for _ in input]

        @staticmethod
        def name():
            return "test-ef-search"

        @staticmethod
        def is_legacy():
            return True

        def embed_query(self, input):
            return self(input)

    orch = object.__new__(server.KnowledgeOrchestrator)
    orch.embed_fn = _Ef()
    orch.chroma_client = chromadb.EphemeralClient()
    monkeypatch.setattr(server.config, "ef_search", 400)
    monkeypatch.setattr(server.config, "collection_name", "ef-" + uuid.uuid4().hex)

    try:
        collection = orch._open_collection()
        assert collection.configuration_json["hnsw"]["ef_search"] == 400
    finally:
        orch.chroma_client.delete_collection(server.config.collection_name)


# ------------------------------------------------------------- self_recall


def test_given_distinct_vectors_when_getting_stats_then_self_recall_is_one(isolated_orchestrator):
    orch = isolated_orchestrator
    orch.collection.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    )

    recall = orch.get_stats()["self_recall"]

    assert recall == {"sampled": 3, "top1": 1.0}


def test_given_identical_vectors_when_getting_stats_then_ties_count_as_hits(isolated_orchestrator):
    orch = isolated_orchestrator
    # "a" and "b" share the exact same vector: whichever one chroma returns as
    # top-1 for that vector, the tie rule must count it as a hit.
    orch.collection.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )

    recall = orch.get_stats()["self_recall"]

    assert recall == {"sampled": 3, "top1": 1.0}


def test_given_empty_collection_when_getting_stats_then_self_recall_reports_empty(isolated_orchestrator):
    assert isolated_orchestrator.get_stats()["self_recall"] == {"sampled": 0, "top1": None}


def test_given_a_miss_when_computing_self_recall_then_fraction_matches():
    from mcp_server import server

    collection = MagicMock()
    collection.count.return_value = 3
    collection.get.side_effect = [
        {"ids": ["x1", "x2", "x3"]},
        {"ids": ["x1", "x2", "x3"], "embeddings": [[1.0], [2.0], [3.0]]},
    ]
    # x1 hits by id, x2 hits via exact-distance tie under a different id, x9 misses.
    collection.query.return_value = {
        "ids": [["x1"], ["x1b"], ["x9"]],
        "distances": [[0.0], [0.0], [7.0]],
    }
    orch = object.__new__(server.KnowledgeOrchestrator)
    orch.collection = collection

    result = orch._compute_self_recall()

    assert result == {"sampled": 3, "top1": round(2 / 3, 4)}


def test_given_two_calls_within_ttl_when_getting_stats_then_probe_runs_once(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    orch.collection.add(ids=["a"], embeddings=[[1.0, 0.0]])
    runs = []
    original = orch._compute_self_recall

    def counting():
        runs.append(1)
        return original()

    monkeypatch.setattr(orch, "_compute_self_recall", counting)

    assert orch.get_stats()["self_recall"]["top1"] == 1.0
    assert orch.get_stats()["self_recall"]["top1"] == 1.0
    assert len(runs) == 1

    orch._invalidate_caches()
    orch.get_stats()
    assert len(runs) == 2


def test_given_expired_cache_when_getting_stats_then_probe_reruns(isolated_orchestrator):
    orch = isolated_orchestrator
    orch.collection.add(ids=["a"], embeddings=[[1.0, 0.0]])
    orch._self_recall_cache = (0.0, {"sampled": 5, "top1": 0.5})

    recall = orch.get_stats()["self_recall"]

    assert recall == {"sampled": 1, "top1": 1.0}


def test_given_reindex_active_without_cache_when_getting_stats_then_probe_is_skipped(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    orch._reindex_progress = {"active": True}

    def forbidden():
        raise AssertionError("probe must not run while reindex is active")

    monkeypatch.setattr(orch, "_compute_self_recall", forbidden)

    assert orch.get_stats()["self_recall"] == {"sampled": 0, "top1": None}


def test_given_reindex_active_with_cache_when_getting_stats_then_last_value_is_returned(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    orch._reindex_progress = {"active": True}
    orch._self_recall_cache = (0.0, {"sampled": 300, "top1": 0.9})
    monkeypatch.setattr(orch, "_compute_self_recall", lambda: (_ for _ in ()).throw(AssertionError()))

    assert orch.get_stats()["self_recall"] == {"sampled": 300, "top1": 0.9}


def test_given_probe_failure_when_getting_stats_then_error_field_reports_it(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator

    def boom():
        raise RuntimeError("x" * 300)

    monkeypatch.setattr(orch, "_compute_self_recall", boom)

    recall = orch.get_stats()["self_recall"]

    assert recall["sampled"] == 0
    assert recall["top1"] is None
    assert recall["error"] == "x" * 200  # truncated to 200 chars


def test_given_a_write_when_add_document_then_self_recall_cache_is_dropped(isolated_orchestrator):
    orch = isolated_orchestrator
    orch.collection.add(ids=["a"], embeddings=[[1.0, 0.0]])
    orch.get_stats()  # fills the probe cache
    assert orch._self_recall_cache is not None

    orch.add_document_from_content("hello world", "new.md", "general")

    assert orch._self_recall_cache is None


def test_given_default_collection_when_reading_effective_ef_search_then_it_is_100(isolated_orchestrator):
    assert isolated_orchestrator._effective_ef_search() == 100


def test_given_index_stats_tool_when_called_then_payload_includes_self_recall(isolated_orchestrator, monkeypatch):
    from mcp_server import server

    monkeypatch.setattr(server, "get_orchestrator", lambda: isolated_orchestrator)

    payload = json.loads(server.get_index_stats())

    assert payload["status"] == "success"
    assert {"sampled", "top1"} <= set(payload["stats"]["self_recall"])
    assert "hnsw_ef_search" in payload["stats"]
