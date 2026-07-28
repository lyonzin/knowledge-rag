# OpenSSF Best Practices — knowledge-rag

**Self-assessment** against the OpenSSF Best Practices Badge criteria at
https://bestpractices.coreinfrastructure.org/en/criteria

**Status:** Ready for **Passing** submission. This document is the evidence pack for the reviewer; the badge itself is registered separately at the URL above and updated as criteria evolve.

**Author:** Ailton Rocha (Lyon.)
**Date:** 2026-07-27
**Version:** v4.6.0

---

## 1. Passing Level (mandatory)

The Passing level is the minimum badge and requires ~15 criteria across Basics, Change Control, Reporting, Quality, Security, and Analysis. Every item here maps to a concrete artefact in this repository.

### 1.1 Basics

| Criterion | Evidence |
|---|---|
| **Project website** (public) | `README.md` at repo root + https://github.com/lyonzin/knowledge-rag |
| **Website describes what it does** | First 30 lines of `README.md` |
| **Interact with developers** | GitHub Issues + Discussions enabled; see `.github/ISSUE_TEMPLATE/config.yml` |
| **English support** | All docs and code comments are English |
| **License is OSI-approved** | `LICENSE` (MIT) |
| **License in known location** | `LICENSE` at repo root; also referenced in `pyproject.toml` `[project]` |
| **Documentation — basics** | `README.md`, `docs/architecture.md`, `docs/usage.md`, `docs/api-reference.md`, `docs/installation.md`, `docs/configuration.md`, `docs/troubleshooting.md` |
| **Documentation — interface** | `docs/api-reference.md` documents all 13 MCP tools with signatures + examples |
| **Discussion of contribution** | `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS` |

### 1.2 Change Control

| Criterion | Evidence |
|---|---|
| **Public VCS** | GitHub public repo |
| **Unique version numbers** | SemVer — `mcp_server.__version__` is the single source of truth; `scripts/check_version_sync.py` fails the pre-commit hook if `pyproject.toml`, `mcp_server/__init__.py`, and `npm/package.json` drift |
| **Release notes** | `CHANGELOG.md` — Keep-a-Changelog format, gated by Quality Gate (a PR that changes runtime code without a CHANGELOG entry fails CI) |
| **Release notes identify fixed vulnerabilities** | `CHANGELOG.md` v4.6.0 explicitly names the CWE classes closed (CWE-22, CWE-59, CWE-287, OWASP LLM01:2025) |

### 1.3 Reporting

| Criterion | Evidence |
|---|---|
| **Bug reporting process** | `.github/ISSUE_TEMPLATE/bug_report.yml` + `feature_request.yml` |
| **Reporting responses** | Maintainer responds on Issues; SLAs documented in `SECURITY.md` for security-class reports |
| **Vulnerability report process** | `SECURITY.md` — GHSA + email fallback |
| **Vulnerability response window** | 48h acknowledgement documented in `SECURITY.md`; 90-day coordinated disclosure |
| **Vulnerability report private channel** | GitHub Security Advisory (private by default) |

### 1.4 Quality

| Criterion | Evidence |
|---|---|
| **Working build system** | `pyproject.toml` + `hatchling` — `python -m build` produces reproducible wheel/sdist |
| **Automated test suite** | `pytest` — currently **~1000 tests** (baseline tracked in `.github/test-count-baseline.txt`, enforced by Quality Gate) |
| **New functionality adds tests** | Quality Gate fails a PR whose runtime diff does not raise the test count or which shrinks it without the `skip-test-count` label |
| **Test coverage measured** | `pyproject.toml` `[tool.coverage.report] fail_under = 35` (baseline; roadmap target 50+ for Phase 4+) |
| **Warnings flag on** | `ruff check`, `mypy --strict`, and `bandit --severity-level high` all run on every PR |
| **Warnings addressed** | Blocking — a warning fails the Quality Gate; no `# noqa` without an inline justification |

### 1.5 Security

| Criterion | Evidence |
|---|---|
| **Developers know secure design** | Maintainer certifications (OSEE — see `~/.claude/CLAUDE.md`); Fase 1 hardening ADR (`docs/adr/0001-fase1-security-hardening.md`) applies STRIDE-style threat modelling |
| **Use good cryptographic practices** | `hmac.compare_digest` for token comparison; `hashlib.sha256` for content integrity; `secrets` module for token generation. No custom crypto |
| **Secured delivery against MITM** | PyPI + npm registries use HTTPS + package signature (Sigstore for PyPI via Trusted Publishing); GHCR images use TLS |
| **Publicly-known vulnerabilities fixed** | `pip-audit --strict` runs weekly + on every PR; `dependabot` opens PRs for CVE-affected deps |
| **No unpatched vulnerabilities of medium+ severity older than 60 days** | Enforced by weekly `pip-audit`; CHANGELOG entries name every closed CVE-class |

### 1.6 Analysis

| Criterion | Evidence |
|---|---|
| **Static code analysis** | `bandit` (SAST) + `semgrep` (`--config=auto`) on every PR — see `.github/workflows/quality-gate.yml` Pillar 1 |
| **Static analysis for common vulnerabilities** | `CodeQL` weekly (`.github/workflows/security.yml`) — Python security queries |
| **Dynamic code analysis (recommended)** | Hypothesis property tests exercise the security helpers; `pytest` integration tests hit real ChromaDB. Coverage-guided fuzzing is a Gold-tier target (see below) |

**Passing status: READY.** Every mandatory criterion has a concrete artefact.

---

## 2. Silver Level (additional criteria)

Silver adds ~30–40 criteria on top of Passing. Items marked ✅ are already in place; items marked ⚠️ are partial; items marked ⏳ are on the roadmap.

### 2.1 Basics — Silver

- ✅ **Distinct roles** — `CODEOWNERS` names the maintainer; `CONTRIBUTING.md` documents reviewer expectations
- ✅ **2FA required for committers** — GitHub org policy (enforced on `lyonzin` account)
- ✅ **DCO/CLA** — commits carry `Signed-off-by` where required; MIT license makes CLA unnecessary
- ⏳ **Roles are documented** in `MAINTAINERS.md` — partial; expand in Phase 5

### 2.2 Change Control — Silver

- ✅ **Repository VCS is a distributed VCS** — Git
- ✅ **Version numbering follows semver rigorously** — Quality Gate `version-sync` hook + `check_version_sync.py`
- ✅ **All non-trivial changes go through review** — branch protection on `master` requires PR + 1 approving review

### 2.3 Reporting — Silver

- ✅ **Vulnerability reports acknowledged within 14 days** — `SECURITY.md` commits to **48 hours** (better than requirement)
- ✅ **Vulnerabilities fixed timeline documented** — see severity table in `SECURITY.md`

### 2.4 Quality — Silver

- ✅ **Reproducible builds** — `hatchling` produces deterministic wheels (no `__pycache__` in sdist, sorted file lists)
- ✅ **Continuous integration** — `.github/workflows/ci.yml` (9-cell OS × Python matrix: Linux + Windows + macOS × 3.11 + 3.12 + 3.13)
- ✅ **Test policy explicit** — `docs/adr/` records testing standards
- ✅ **Automated regression tests** — every closed CVE-class has a regression test in `tests/security/`
- ⚠️ **Test coverage >80%** — currently gated at 35% baseline, roadmap target 50+ in Phase 4 and 80+ in Phase 5+
- ✅ **Style guide enforced** — `ruff format` + `ruff check` on every PR

### 2.5 Security — Silver

- ✅ **Documented threat model** — `SECURITY.md` § Threat Model + `docs/adr/0001-fase1-security-hardening.md`
- ✅ **Uses good cryptographic practices** — see Passing 1.5 (documented per-primitive)
- ✅ **Uses SAST tools** — `bandit` + `semgrep` + `CodeQL`
- ✅ **Uses SCA tools** — `pip-audit` + `dependabot`
- ✅ **Signed releases** — PyPI OIDC Trusted Publishing (Sigstore-backed); npm publish uses `NODE_AUTH_TOKEN` on GHA (short-lived); git tags are signed by the release author
- ✅ **Secrets scanning** — `gitleaks` on every PR + full-history scan
- ✅ **Hardened dependency graph** — `pyproject.toml` version pins + `dependabot` grouped updates
- ✅ **Memory-safe primary language** — Python (Rust for hot paths in `fastembed` via ONNX Runtime)

### 2.6 Analysis — Silver

- ✅ **Static analysis on major public release** — Quality Gate is blocking on every PR that touches `master`
- ✅ **Dynamic analysis in CI** — `hypothesis` property tests + full integration suite
- ✅ **Dynamic analysis for memory safety** — Python primary; the C extension surface (ONNX Runtime, ChromaDB SQLite) is upstream-audited

**Silver status: LIKELY READY** pending coverage climb to 80% and MAINTAINERS.md expansion.

---

## 3. Gold Level (aspirational)

Gold adds ~15 more criteria beyond Silver. The following are **explicit roadmap items** rather than current-state claims.

- ⏳ **SBOM published per release** — target Phase 5; will publish CycloneDX + SPDX alongside PyPI/npm release assets
- ⏳ **Reproducible builds verified externally** — target Phase 5; will document exact `hatchling` invocation and publish `.buildinfo`-style attestations
- ⏳ **Coverage-guided fuzzing in CI** — Hypothesis property tests exist; will add `atheris` (Python libFuzzer) coverage-guided fuzzer for parsers and `security.py`
- ⏳ **Multiple maintainers** — currently single-maintainer; onboarding a second maintainer with commit rights is a Phase 6 target
- ⏳ **Publicly documented governance model** — expand `CONTRIBUTING.md` into a full `GOVERNANCE.md`
- ⏳ **All CI runs on hardened runners** — GitHub-hosted runners are baseline; self-hosted `harden-runner` action is a Phase 5 target
- ⏳ **All CI actions pinned by SHA** — currently pinned by tag; `dependabot` `github-actions` ecosystem opens updates; migrating to SHA pins is a Phase 5 target

**Gold status: ROADMAPPED.**

---

## 4. Registration

The self-cert URL will be:

```
https://bestpractices.coreinfrastructure.org/projects/XXXX
```

Where `XXXX` is assigned on submission. Update the placeholder in `README.md` after the badge is claimed. The badge Markdown lives in `README.md` alongside the CI badges.

## 5. Maintenance

This document is reviewed:

- **On every minor release** — verify criteria still hold.
- **When adding a new dependency** — reassess SCA coverage.
- **When the OpenSSF criteria change** — currently reviewed annually against the upstream criteria list.
