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
| Path traversal in document CRUD and retrieval tools | CWE-22 | `security.validate_path_within(base, candidate)` resolves the candidate against the documents directory and checks containment. Traversal and absolute paths outside that directory, NUL bytes, and NTFS alternate data streams are rejected. Absolute paths inside the corpus are supported. There is no public `add_document_from_file` MCP tool. | v4.6.0 |
| Symlink escape or directory cycles during discovery | CWE-59 | Check resolved walk roots, child directories, and file candidates against the corpus root; track visited directories to terminate cycles. Links contained within the corpus remain supported. | Current unreleased audit fixes; the previous v4.6.0 claim was not backed by active integration tests |
| Bearer auth was declared but never enforced — operators believed the SSE/HTTP transport was protected when it was not | CWE-287 | Stdlib ASGI middleware (`security.BearerAuthMiddleware`) enforces `Authorization: Bearer <token>` with `hmac.compare_digest` (constant-time). Returns `401 + WWW-Authenticate: Bearer realm="knowledge-rag"` on missing/wrong tokens. `lifespan`/`websocket` scopes pass through. Empty token → `WARN "bearer auth disabled"` (backwards compatible for pre-4.6 users who set the field but had never really been protected). | v4.6.0 |
| Prompt injection through externally sourced content | OWASP **LLM01:2025** | `add_from_url` neutralizes a finite set of known control-token sentinels before wrapping the stored document in a provenance fence. Parsing preserves detected provenance on documents and chunks. Ordinary files without that marker are not automatically rewritten. These measures do not reliably detect arbitrary instructions or make retrieved text trustworthy; the consuming client must treat retrieved content as data. | URL sanitization v4.6.0; parser propagation verified by the current audit |
| Sensitive data indexed by accident | (design — allowlist) | Discovery uses configured supported extensions/filenames and exclusion patterns. This is not a secret scanner: a supported `.json`, `.yaml`, `.txt`, or hidden `.md` file can contain credentials. Keep sensitive material outside the corpus or explicitly exclude it, and do not publish index data. | v4.0+ |
| Rate limiting on MCP endpoints | CWE-770 (partial) | A sliding-window counter in `mcp_server/ratelimit.py` bounds tool calls to `requests_per_minute + burst` in the window. All clients currently share the same `default` bucket. Disabled by default. | v4.0+ |
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
| Latest published 4.10.x release | Current maintenance line; check release notes for the fixes actually included |
| Unreleased branches / pull requests | Under review; not a released security guarantee |
| Previous minor release | Security patch window of 30 days after the next minor release, under the policy below |
| Older releases | Unsupported; upgrade to the maintained release |

When a new minor version ships, the previous minor gets a final 30-day security patch window and is then unsupported. The version embedded in a development branch does not prove that its fixes have been published. Consult [releases](https://github.com/lyonzin/knowledge-rag/releases) and the associated changelog.

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security concerns.**

**Please report via GitHub Security Advisories (private, native to GitHub):**

**→ https://github.com/lyonzin/knowledge-rag/security/advisories/new**

Use the private reporting form if it is available to your account. Reports are handled in GitHub's security workflow. No external email address is used for reporting.

If you cannot use the GitHub Security Advisory form (rare), open a public
issue with the label `security-triage` **describing only that you have a
security concern and how to reach you privately** — do not disclose details
publicly. The maintainer will contact you to move the report off-issue.

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
- Release publishing is triggered by the repository's release workflow. Do not infer signed commits or verified signatures merely from a successful build; inspect the specific release and its available provenance.

---

## Security Best Practices for Users

- **Do not commit `data/`** — the ChromaDB indices contain the raw text of everything you indexed.
- **Do not commit `documents/`** if it holds sensitive material — check your `.gitignore`.
- **Rotate `server.auth.bearer_token`** if it was ever pasted into a shell history or shared config.
- **Enable rate limiting** for any deployment reachable off `localhost` (`server.rate_limit.enabled: true` in `config.yaml`).
- **Review `add_from_url` sources** — token neutralization is a limited mitigation, not a prompt-injection prevention guarantee. HTTP(S) URL fetching is not a network-isolation boundary.
- **Prefer `stdio` transport** for single-user local usage. For a network deployment, set `server.auth.bearer_token` and apply appropriate TLS and access controls.
- **Treat the metrics endpoint separately** — it currently binds to `0.0.0.0` and does not inherit MCP bearer authentication. Enable and expose it deliberately.
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

Security claims should be supported by active regression tests and released fixes. Expected-failure tests and planned defenses are not equivalent to a shipped mitigation.
