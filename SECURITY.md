# Security Policy

`knowledge-rag` is a local RAG server. It indexes content the operator did not necessarily write (fetched URLs, downloaded PDFs, synced GitHub repos) and it exposes an MCP surface that is driven by an LLM client. That combination — untrusted content, untrusted client, local privileges — is what this document takes seriously.

---

## Threat Model

`knowledge-rag` runs **locally**. The attacker model assumes:

- The **user is trusted** — they own the machine.
- The **MCP client** (Claude Code, Claude Desktop, Cursor, etc.) is trusted as software, but its *inputs* to the server are not.
- **UNTRUSTED — indexed documents**: content fetched by `add_from_url`, PDFs/DOCX/HTML dropped into `documents/`, GitHub repos synced by the operator. Any of those may be attacker-influenced.
- **UNTRUSTED — MCP client requests**: paths, queries, filters. An LLM client can be talked into passing hostile arguments; the server must not trust them.
- **UNTRUSTED — network responses** when `add_from_url` fetches a URL.

Anything requiring the attacker to already own the local filesystem is **out of scope** — that is the trust model of a local RAG.

### Attack Surface × Mitigations

| Vector | CWE / OWASP | Mitigation | Since |
|---|---|---|---|
| Path traversal in `add_document`, `update_document`, `remove_document`, `get_document`, `search_similar`, `add_document_from_file` | CWE-22 | `security.validate_path_within(base, candidate)` — `Path.resolve()` + `is_relative_to()`. Rejects `..`, absolute paths, NUL bytes, NTFS alternate data streams. Applied to **every** CRUD tool. | v4.6.0 |
| Symlink escape via `os.walk(followlinks=True)` | CWE-59 | Every walk yield **and** every file candidate is containment-checked against the base with a `Path.is_symlink()` fast-path `lstat` so the non-symlink case costs one syscall. `followlinks=True` is preserved — option (a) was rejected because `open()` follows symlinks regardless. | v4.6.0 |
| Bearer auth was declared but never enforced — operators believed the SSE/HTTP transport was protected when it was not | CWE-287 | Stdlib ASGI middleware (`security.BearerAuthMiddleware`) enforces `Authorization: Bearer <token>` with `hmac.compare_digest` (constant-time). Returns `401 + WWW-Authenticate: Bearer realm="knowledge-rag"` on missing/wrong tokens. `lifespan`/`websocket` scopes pass through. Empty token → `WARN "bearer auth disabled"` (backwards compatible for pre-4.6 users who set the field but had never really been protected). | v4.6.0 |
| Prompt injection via externally sourced content (`add_from_url`, PDFs, DOCX, HTML) — retrieved text carries model control tokens or "ignore previous instructions" framing, and the *consuming* LLM executes it | OWASP **LLM01:2025** | Three-layer defense in `mcp_server/security.py`: **(1)** neutralize known injection sentinels (`<\|im_start\|>`, `<\|system\|>`, `[INST]`, `<<SYS>>`, `### system:`, forged `</external_content>`) by inserting `​` after the first character — idempotent, human-readable, tokenizer-broken; **(2)** wrap chunks in `<external_content source="..." sha256="...">…</external_content>` so provenance survives restart + reindex; **(3)** emit `external_source: true` + `content_hash` on retrieval so the calling LLM can weight external content differently. **Ordering is pinned by a regression test** — neutralize **before** wrap, otherwise a payload can ship `</external_content>` and escape its own fence. | v4.6.0 |
| Sensitive data indexed by accident (`.env`, credentials files) | (design — whitelist) | Ingestion uses an explicit allowlist (`_SUPPORTED_SUFFIXES` in `mcp_server/config.py`). `.env`, `.pem`, `.key`, dotfiles, and anything else outside the 18 supported suffixes are **never** enumerated by the walker. User awareness in docs — do not commit `data/` (ChromaDB indices may contain sensitive text) or `documents/` if it contains sensitive material. | v4.0+ |
| Rate limiting on MCP endpoints | CWE-770 (partial) | Token-bucket limiter (`mcp_server/ratelimit.py`) — configurable `requests_per_minute` + `burst`. Off by default; recommended on for public-facing SSE/HTTP transports. Applied via the `@rate_limited` decorator on MCP tools. | v4.0+ |
| Supply chain — PyPI publish credential theft | CWE-1104 | **PyPI Trusted Publishing via OIDC** (`id-token: write` in `.github/workflows/release.yml`, `pypa/gh-action-pypi-publish@release/v1`). No long-lived API tokens on GitHub. | v4.0+ |
| Deserialization of untrusted input | CWE-502 | Only `json.loads` and `yaml.safe_load` are used (`grep -rE "import pickle|yaml\.load\b"` returns empty). `pickle`, `yaml.load` (unsafe), `shelve`, and `marshal` are all absent. | v4.0+ |
| Command injection via tool arguments | CWE-78 | No `os.system` / `subprocess.run(shell=True)` on user-controlled strings anywhere in `mcp_server/`. Verified by `bandit` (`B605`, `B602`) on every PR. | v4.0+ |

**Non-goals** (deliberately unmitigated):
- Denial of service via genuinely large documents the operator chose to index. Ingestion is bounded by disk, not by hard file-size caps.
- Any attack requiring pre-existing filesystem access.
- Adversarial ML attacks on the embedding model itself (model poisoning, embedding inversion) — the model is a trusted upstream artifact.

---

## Supported Versions

| Version | Status |
|---|---|
| **4.6.x** | ✅ Active — Phase 1 security hardening baseline |
| 4.5.x | ⚠️ Security-only patches for **30 days** after 4.6.0 |
| 4.4.x | ❌ EOL — upgrade to 4.6 recommended |
| < 4.0 | ❌ EOL |

When a new minor version ships, the previous minor gets one final security patch window and is then unsupported.

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security concerns.**

Use one of these private channels:

1. **GitHub Security Advisory** (preferred):
   https://github.com/lyonzin/knowledge-rag/security/advisories/new
2. **Email**: lyonzin@users.noreply.github.com

When you report, include:

- A clear description of the issue and its impact.
- Reproduction steps or a proof-of-concept.
- The version of `knowledge-rag` affected (`pip show knowledge-rag`).
- Operating system and Python version.
- Whether you have already disclosed the issue elsewhere.

### What to expect

- **Acknowledgement**: within **48 hours**.
- **Initial assessment**: within 5 business days.
- **Patch + coordinated disclosure**: within **90 days** for CRITICAL / HIGH severity, extendable by mutual agreement.
- **Public advisory** with CVE assignment when applicable, credit to the reporter (unless anonymity is requested).

### Severity & patch timelines

| Severity | Description | Target patch |
|---|---|---|
| Critical | Remote impact without authentication, or data corruption | 7 days |
| High     | Authenticated remote impact, or local impact with privilege | 14 days |
| Medium   | Limited impact requiring specific conditions | 30 days |
| Low      | Minor information disclosure or hardening | 60 days or next release |

---

## Security Automation in CI

Every PR against `master` must pass **Pillar 1** of the Quality Gate before merge:

- **`bandit`** — Python SAST (`--severity-level high --confidence-level medium`), runs on `mcp_server/` + `scripts/`.
- **`semgrep`** — `--config=auto`, runs on `mcp_server/` + `scripts/`.
- **`pip-audit --strict`** — CVE scan against the fully resolved dependency graph (`pip download` first, then audit).
- **`gitleaks`** — secret scan across full history with `.github/gitleaks.toml`.

Weekly scheduled jobs (`.github/workflows/security.yml`, Monday 06:00 UTC):

- **`CodeQL`** — GitHub-native SAST for Python.
- **`pip-audit`** — repeated in schedule so newly-disclosed CVEs in already-pinned versions surface without waiting for a PR.

Automated dependency updates:

- **`dependabot`** monitors `pip` (repo root), `github-actions`, `npm` (`/npm`), and `docker` (repo root). See `.github/dependabot.yml` for exact cadence and update-type policy.

Release integrity:

- **PyPI Trusted Publishing** (OIDC) — no long-lived API tokens on GitHub.
- Release commits are signed by the tag creator; the release workflow verifies the ref before publishing.

---

## Security Best Practices for Users

- **Do not commit `data/`** — the ChromaDB indices contain the raw text of everything you indexed.
- **Do not commit `documents/`** if it holds sensitive material — check your `.gitignore`.
- **Rotate `auth_bearer_token`** if it was ever pasted into a shell history or shared config.
- **Enable rate limiting** for any deployment reachable off `localhost` (`server.rate_limit.enabled: true` in `config.yaml`).
- **Review `add_from_url` sources** — `knowledge-rag` defends against prompt injection in retrieved content, but it should not be pointed at hostile domains indiscriminately.
- **Prefer `stdio` transport** for single-user local usage. Use `sse` or `streamable-http` only with `auth_bearer_token` set and rate limiting enabled.
- **Do not disable TLS verification** on the fetch path — the URL loader defaults to strict verification for a reason.

---

## Scope

**In scope:**
- The `mcp_server` Python package.
- The NPM CLI wrapper at `npm/`.
- The Docker image `ghcr.io/lyonzin/knowledge-rag`.
- Release pipeline workflows under `.github/workflows/`.

**Out of scope:**
- Issues in upstream dependencies (please report to those projects; we track advisories via `dependabot` + `pip-audit`).
- Findings only achievable with attacker-controlled access to the local filesystem — that is the trust model.
- Denial of service via genuinely large input the user opted to index.
- Issues already covered by public dependency scanners (Socket, Snyk, CodeQL).

---

## Coordinated Disclosure

We follow a **90-day coordinated disclosure** window from acknowledgement to public disclosure, extendable by mutual agreement if the fix needs more time. After the fix ships, we publish a GitHub Security Advisory with credit to the reporter unless anonymity is requested.

## Hall of Fame

Reporters who help improve `knowledge-rag` security are acknowledged in [Security Advisories](https://github.com/lyonzin/knowledge-rag/security/advisories) and in the README.

## Bug Bounty

This project does not currently offer a paid bug bounty. Volunteer contributions to security are deeply appreciated and credited publicly.

---

Thank you for helping keep `knowledge-rag` and its 70+ enterprise users safe.
