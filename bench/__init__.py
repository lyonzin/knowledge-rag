"""Performance benchmarks for knowledge-rag.

Pillar 5: Scalability — measure index throughput, search latency,
memory under load, and concurrent throughput across releases.

Run locally with:
    pytest bench/ -v --benchmark-only --benchmark-json=bench-results.json

CI runs the same twice on the PR branch and twice on master, interleaved,
then feeds the four JSON files to scripts/check_perf_regression.py for
the 10% gate.
"""
