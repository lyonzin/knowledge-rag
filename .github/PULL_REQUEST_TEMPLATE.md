<!--
Thank you for contributing to knowledge-rag.

Every PR is evaluated against the 7 Pillars Quality Gate. Please fill every
section. Use [N/A] with a one-line justification if a section truly does not
apply.

See CONTRIBUTING.md for full details.
-->

## Summary

<!-- One paragraph: what changes and why. Link to the issue this fixes. -->

Closes #

## Type of change

- [ ] feat — new feature
- [ ] fix — bug fix
- [ ] docs — documentation only
- [ ] refactor — no behavior change
- [ ] perf — performance improvement
- [ ] test — adding or improving tests
- [ ] chore — tooling, deps, CI
- [ ] BREAKING CHANGE (explain in Migration section below)

## What changed

<!-- Bullet list of the actual changes. File paths welcome. -->

-
-

## Why

<!-- The motivation. What problem does this solve? Why this approach over alternatives? -->

---

## 7 Pillars Quality Gate

> Mark each item. CI enforces these via the `quality-gate.yml` workflow.

### 1. Security

- [ ] No new secrets, tokens, or credentials in the diff (gitleaks will block)
- [ ] No new use of `eval`, `exec`, `subprocess shell=True`, `pickle.loads` on untrusted input, or arbitrary deserialization
- [ ] New dependencies (if any) reviewed for known CVEs and license compatibility
- [ ] Path traversal, command injection, and SSRF surfaces explicitly considered for any new I/O code

### 2. Stability

- [ ] All existing tests still pass on Linux + Windows × Python 3.11/3.12
- [ ] New behavior covered by tests; tests are deterministic (no `time.sleep` / network / OS-scheduler dependencies)
- [ ] Coverage does not regress (codecov gate)
- [ ] No tests were skipped, deleted, or marked `xfail` to make the PR pass

### 3. Memory leak

- [ ] Long-lived objects (orchestrator, watcher, cache) are bounded
- [ ] New caches have eviction policy (LRU, TTL, or explicit size limit)
- [ ] No new global state that grows unbounded with usage
- [ ] If you added a new module that loads heavy resources, consider lazy initialization

### 4. Versatility

- [ ] Works on Linux, Windows, macOS (paths, line endings, locale considered)
- [ ] Works on Python 3.11, 3.12, 3.13 (no Python-version-specific syntax without fallback)
- [ ] No hardcoded paths, locales, or encodings (use `pathlib.Path`, `encoding="utf-8"` explicit)
- [ ] If you touched a parser, all 20 supported formats still parse correctly

### 5. Scalability

- [ ] No O(n²) or worse algorithms on user-controlled inputs
- [ ] Benchmark impact considered (run `pytest bench/` locally if you touched search/index/embed)
- [ ] If perf regression > 10% in any metric, justification provided below
- [ ] Concurrency safety: no new shared mutable state without lock or documented thread confinement

**Performance impact** (required if you touched `mcp_server/server.py`, `mcp_server/ingestion.py`, or `bench/`):

```
metric          before    after    delta
search p95      ___ ms    ___ ms   ___%
index docs/sec  ___       ___      ___%
RSS @ 1k docs   ___ MB    ___ MB   ___%
```

### 6. Versioning

- [ ] If this is user-facing change: bumped version in `pyproject.toml`, `mcp_server/__init__.py`, and `npm/package.json` atomically
- [ ] If this is a breaking change: bumped MAJOR, added migration notes in CHANGELOG, marked `BREAKING CHANGE:` in commit footer
- [ ] CHANGELOG updated with entry under `## Unreleased` in README.md
- [ ] Public API surface (`mcp_server/server.py` MCP tool decorators) unchanged, OR breaking changes documented

### 7. Quality

- [ ] `ruff check` passes
- [ ] `ruff format --check` passes
- [ ] Type hints on new public functions (`mypy --strict` clean for new files)
- [ ] Docstrings on new public functions (used by `interrogate`)
- [ ] Cyclomatic complexity reasonable (`radon cc --max=C`)
- [ ] No dead code (`vulture` would not flag new code)
- [ ] PR is reasonably sized (< 500 lines of diff preferred; bigger PRs split or justify)

---

## Migration / Breaking changes

<!-- Required if you marked BREAKING CHANGE. Otherwise write N/A. -->

N/A

## Test plan

<!-- How will the reviewer verify this works? What did you actually run? -->

- [ ] `pytest tests/ -v` passed locally
- [ ] `pre-commit run --all-files` clean
- [ ] Manual smoke test: <!-- describe what you did -->

## Documentation

- [ ] Updated `README.md` (if user-facing)
- [ ] Updated `docs/` (if applicable)
- [ ] Added entry to `## Unreleased` in README CHANGELOG section

## Reviewer checklist

<!-- Do not edit. The reviewer fills this. -->

- [ ] Reviewed line-by-line
- [ ] Verified the 7 pillars CI status checks are green
- [ ] Verified no obvious adversarial implications
- [ ] Approved performance impact

---

By submitting this PR I confirm I read [CONTRIBUTING.md](../CONTRIBUTING.md) and agree to the [Code of Conduct](../CODE_OF_CONDUCT.md).
