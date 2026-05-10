"""Chaos suite — failure injection tests.

Each test in this directory simulates a real-world failure mode that
once produced (or could plausibly produce) a silent corruption or
"frozen MCP server" symptom in production. The chaos suite runs in
.github/workflows/nightly.yml so it does not slow regular PRs but
still catches resilience regressions weekly.

Markers:
- ``@pytest.mark.chaos`` — only collected by the nightly job
- Each test must clean up after itself; chaos tests run in shared CI runners

Failure modes exercised:
- HuggingFace Hub returning 503 mid-download
- ChromaDB SQLite truncated mid-write
- Disk full during indexing
- Watchdog observer crash (FS unavailable)
- ONNX cache file zeroed (the exact bug from v3.8.1)
"""
