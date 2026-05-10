"""Performance benchmarks for knowledge-rag.

Pillar 5: Scalability — measure index throughput, search latency,
memory under load, and concurrent throughput across releases.

Run locally with:
    pytest bench/ -v --benchmark-only --benchmark-json=bench-results.json

CI runs the same against master and against the PR branch, then feeds
both JSON files to scripts/check_perf_regression.py for the 10% gate.
"""
