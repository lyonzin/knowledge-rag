# Captured from a knowledge-rag v3.6.0 user setup. Used by
# tests/test_backwards_compat.py to ensure newer Config classes still
# load this exact shape without exceptions.
#
# Do not modify — this is a frozen snapshot for regression testing.

models:
  embedding:
    name: BAAI/bge-small-en-v1.5
    dim: 384
    gpu: false
  reranker:
    name: Xenova/ms-marco-MiniLM-L-6-v2
    enabled: true

chunking:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100

search:
  hybrid_alpha: 0.3
  max_results: 5
  query_cache_ttl: 300

categories:
  security: ["security/", "documents/security/"]
  ctf: ["ctf/"]
  development: ["dev/", "code/"]
