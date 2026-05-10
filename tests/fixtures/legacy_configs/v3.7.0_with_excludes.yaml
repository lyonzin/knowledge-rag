# Captured from a knowledge-rag v3.7.0 user setup with the exclude_patterns
# feature added in v3.4.0. Used to validate that old configs continue
# loading after each new release.

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

search:
  hybrid_alpha: 0.3
  max_results: 5

exclude_patterns:
  - "*.tmp"
  - "*.bak"
  - "node_modules/"
  - ".venv/"
