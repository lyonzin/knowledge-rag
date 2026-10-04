"""The response limit is not an internal candidate-budget limit."""

from types import SimpleNamespace

from mcp_server import server
from mcp_server.search_candidates import select_rerank_candidates


def test_semantic_candidates_reach_reranker_beyond_public_response_cap(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "max_results", 20)
    monkeypatch.setattr(server.config, "reranker_enabled", True)
    monkeypatch.setattr(server.config, "reranker_top_k_multiplier", 3)
    ids = [f"doc{index}_0" for index in range(30)]
    texts = ["the relevant explanation" if index == 24 else f"unrelated passage {index}" for index in range(30)]
    orch.collection.add(
        ids=ids,
        documents=texts,
        embeddings=[[1.0 + index * 0.01, 1.0] for index in range(30)],
        metadatas=[{"source": f"{index}.txt"} for index in range(30)],
    )
    seen = []

    def rerank(query, documents, top_k):
        seen.extend(item["chunk_id"] for item in documents)
        for item in documents:
            item["reranker_score"] = float(item["document"] == "the relevant explanation")
        return sorted(documents, key=lambda item: item["reranker_score"], reverse=True)[:top_k]

    orch.reranker = SimpleNamespace(rerank=rerank)
    results = orch.query("explain this concept", max_results=10, hybrid_alpha=1.0)
    assert "doc24_0" in seen
    assert results[0]["source"] == "24.txt"
    assert len(results) == 10


def test_weighted_fusion_preserves_both_channels_before_reranker():
    candidates = []
    for rank in range(1, 31):
        candidates.append(
            (
                f"semantic-{rank}",
                {
                    "rrf_score": 0.3 / (60 + rank) + 0.7 / 1060,
                    "semantic_rank": rank,
                    "bm25_rank": None,
                },
            )
        )
        candidates.append(
            (
                f"keyword-{rank}",
                {
                    "rrf_score": 0.7 / (60 + rank) + 0.3 / 1060,
                    "semantic_rank": None,
                    "bm25_rank": rank,
                },
            )
        )
    old_selection = sorted(candidates, key=lambda row: row[1]["rrf_score"], reverse=True)[:30]
    assert all(identity.startswith("keyword-") for identity, _ in old_selection)
    selected = select_rerank_candidates(candidates, limit=30, results=10)
    identities = {identity for identity, _ in selected}
    assert len(selected) == 30
    assert {f"semantic-{rank}" for rank in range(1, 11)} <= identities
    assert {f"keyword-{rank}" for rank in range(1, 11)} <= identities


def test_one_channel_or_small_union_keeps_ranked_candidates():
    rows = [(str(rank), {"rrf_score": 1 / rank, "semantic_rank": rank, "bm25_rank": None}) for rank in range(1, 60)]
    assert select_rerank_candidates(rows, limit=30, results=10) == rows[:30]
    assert select_rerank_candidates(rows[:5], limit=30, results=10) == rows[:5]
