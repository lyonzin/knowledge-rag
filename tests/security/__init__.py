"""Security regression tests for knowledge-rag (ADR-0001, Bloco A).

Each module here is a proof-of-concept that a specific vulnerability class is
closed:

* ``test_path_traversal``   — CWE-22, sandbox escape via ``filepath``
* ``test_symlink_escape``   — CWE-59, corpus escape via planted symlinks
* ``test_prompt_injection`` — OWASP LLM01:2025, hostile retrieved content
* ``test_bearer_auth``      — CWE-287, unauthenticated HTTP transport
"""
