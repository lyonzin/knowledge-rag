# Contributing to knowledge-rag

Thank you for considering contributing! This project is used daily by 70+ companies and teams. Quality is non-negotiable, but the process should still be welcoming and clear.

This guide explains how to propose changes, what we check, and what to expect from review.

---

## Table of contents

- [Quick start](#quick-start)
- [The 7 Pillars](#the-7-pillars-quality-gate)
- [Local development](#local-development)
- [Pre-commit hooks](#pre-commit-hooks)
- [Running tests](#running-tests)
- [Pull request lifecycle](#pull-request-lifecycle)
- [Commit message convention](#commit-message-convention)
- [Reporting bugs](#reporting-bugs)
- [Reporting security issues](#reporting-security-issues)
- [Code of Conduct](#code-of-conduct)

---

## Quick start

```bash
git clone https://github.com/lyonzin/knowledge-rag.git
cd knowledge-rag
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .
pip install -r requirements-dev.txt   # ruff, pytest, mypy, pre-commit
pre-commit install                    # one-time
```

Run the full local check before pushing:

```bash
pre-commit run --all-files
pytest tests/ -v
```

---

## The 7 Pillars (Quality Gate)

Every PR is automatically evaluated against seven pillars. All must pass before manual review by [@lyonzin](https://github.com/lyonzin).

| # | Pillar | What it covers | Tools |
|---|---|---|---|
| 1 | **Security** | SAST, secrets, dependency CVEs, supply chain | bandit, semgrep, gitleaks, pip-audit, Snyk, CodeQL, Socket |
| 2 | **Stability** | Test determinism, coverage trend, flake detection | pytest-rerunfailures, codecov trend gate, test count guard |
| 3 | **Memory Leak** | RSS growth under load, leak isolation | pytest-memray, baseline tests, nightly soak (1h continuous) |
| 4 | **Versatility** | OS × Python matrix, format matrix, locale matrix, preset matrix | matrix CI on Linux/Windows/macOS × 3.11/3.12/3.13 |
| 5 | **Scalability** | Performance regression, concurrent load | bench suite (MRR@5, p50/p95/p99 latency, throughput), perf gate at 10% regression |
| 6 | **Versioning** | Atomic version sync, API surface stability, CHANGELOG enforcement | pre-commit hook, griffe, conventional commits, backwards-compat tests |
| 7 | **Quality** | Style, types, docstrings, complexity, dead code | ruff, mypy strict, interrogate, radon, vulture |

Your PR will show 7 status checks at the bottom — one per pillar. If any is ❌, fix it before requesting review. If you genuinely cannot fix one (e.g., a regression that is intentional), explain why in the PR description and tag `@lyonzin` for an override discussion.

---

## Local development

### Recommended tooling

```bash
# Required
python >= 3.11
pip install -e .

# Development
pip install ruff pytest pytest-cov pytest-rerunfailures pytest-memray
pip install mypy interrogate radon vulture
pip install pre-commit hypothesis

# Optional (only if you touch benchmarks)
pip install pytest-benchmark
```

### Project layout

```
mcp_server/         # Production code
tests/              # All tests (unit + property + integration + memory baseline)
bench/              # Performance benchmarks (run on PRs + nightly)
docs/               # User-facing documentation
examples/           # Sample MCP client configs
.github/workflows/  # CI definitions
```

---

## Pre-commit hooks

`pre-commit` runs lightweight checks before each commit. **Required** — PRs that bypass pre-commit will fail CI on the same checks.

```bash
pre-commit install              # install once
pre-commit run --all-files      # run on full repo (occasional manual run)
```

Hooks enforced:
- `ruff check` and `ruff format --check`
- `version-sync` — `pyproject.toml` / `mcp_server/__init__.py` / `npm/package.json` must agree
- `gitleaks` — block secrets in commits

---

## Running tests

```bash
# Fast: unit + property tests, no model downloads
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=mcp_server --cov-report=term-missing

# Memory baseline (slow, requires pytest-memray)
pytest tests/test_memory_baseline.py -v --memray

# Property-based fuzzing (Hypothesis)
pytest tests/test_ingestion_property.py -v --hypothesis-show-statistics

# Benchmarks (slow, generates artifacts in bench/results/)
pytest bench/ -v --benchmark-only
```

**HuggingFace offline mode is enforced in CI.** If a test triggers a real download it will fail fast. Use ``unittest.mock.patch("mcp_server.server.TextEmbedding")`` and ``patch("mcp_server.server.TextCrossEncoder")``.

---

## Pull request lifecycle

1. **Fork + branch** — `feat/<short-desc>` for features, `fix/<issue>-<desc>` for bugs, `docs/<area>` for docs, `chore/<topic>` for chores.
2. **Open PR early as draft** — visibility helps avoid duplicate work.
3. **Fill the PR template** — every checkbox in the template is enforced. Mark `N/A` and explain if a section truly does not apply.
4. **Pass the 7 Pillars** — green checks across the board.
5. **Wait for review** — [@lyonzin](https://github.com/lyonzin) reviews every PR line by line, even after green CI. Response time target: 48h on weekdays.
6. **Address feedback** — push more commits, do not force-push during review (preserves discussion thread).
7. **Squash merge** — reviewer (or you with permission) squashes with a clean conventional commit message.

### What gets a fast track
- Pure docs / typo fixes
- New tests for existing behavior
- Backwards-compatible bug fixes with regression test

### What needs more discussion
- New MCP tools (public API surface)
- Changes to embedding model, reranker model, or chunking strategy
- Anything touching ChromaDB schema or `index_metadata.json` format
- Changes to the 7 Pillars themselves

### What is generally rejected
- Style-only refactors with no behavior change (use ruff)
- Breaking API changes without strong justification + migration path
- New dependencies without explaining why existing options were ruled out
- PRs that disable failing tests instead of fixing them

---

## Commit message convention

We use [Conventional Commits](https://www.conventionalcommits.org/). Enforced via `commitlint` in CI on PR titles and squash messages.

```
<type>(<scope>): <short summary>

<longer body explaining the why>

<footer with refs / breaking-change marker>
```

Allowed types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `build`, `ci`, `style`.

Examples:
- `feat(server): add semantic_alpha config option`
- `fix(embeddings): raise loud on load failures`
- `docs(readme): bump What's New to v3.9.0`
- `perf(search): use BM25 inverted index for 3x speedup`

Breaking changes go in the footer: `BREAKING CHANGE: explanation`.

---

## Reporting bugs

Use the [Bug Report template](https://github.com/lyonzin/knowledge-rag/issues/new?template=bug_report.yml). Required fields:
- Version (`pip show knowledge-rag` output)
- Reproduction steps
- Expected vs actual behavior
- Logs / stack traces (sanitize secrets)

---

## Reporting security issues

**Do not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it.

---

## Questions?

- General usage: [GitHub Discussions](https://github.com/lyonzin/knowledge-rag/discussions)
- Bugs: GitHub Issues with the bug template
- Security: SECURITY.md
- Direct: [@lyonzin](https://github.com/lyonzin) — but prefer public channels so the community benefits

Welcome aboard, and thank you for helping make knowledge-rag better.
