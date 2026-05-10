# Security Policy

We take the security of `knowledge-rag` seriously. This document describes how to report issues responsibly and what to expect in response.

## Supported versions

Only the latest minor release line receives security fixes.

| Version | Supported |
|---------|-----------|
| 3.8.x   | ✅ |
| < 3.8   | ❌ |

When a new minor version ships (e.g. 3.9.0), the previous minor (3.8.x) gets one final security patch and is then unsupported.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security concerns.**

Use one of these private channels:

1. **GitHub Security Advisory** (preferred):
   https://github.com/lyonzin/knowledge-rag/security/advisories/new

2. **Email**: lyonzin@users.noreply.github.com

When you report, include:

- A clear description of the issue and its impact
- Reproduction steps or a proof-of-concept
- The version of `knowledge-rag` affected (`pip show knowledge-rag`)
- Operating system and Python version
- Whether you have already disclosed the issue elsewhere

## What to expect

- **Acknowledgement**: within 48 hours
- **Initial assessment**: within 5 business days
- **Patch timeline**: depends on severity (see below)
- **Public disclosure**: coordinated with the reporter, typically after a fix is released

### Severity & timelines

| Severity | Description | Target patch |
|----------|-------------|--------------|
| Critical | Remote impact without authentication, or data corruption | 7 days |
| High     | Authenticated remote impact, or local impact with privilege | 14 days |
| Medium   | Limited impact requiring specific conditions | 30 days |
| Low      | Minor information disclosure or hardening | 60 days or next release |

## Scope

In scope:
- The `mcp_server` Python package
- The NPM CLI wrapper at `npm/`
- The Docker image `ghcr.io/lyonzin/knowledge-rag`
- Release pipeline workflows under `.github/workflows/`

Out of scope:
- Issues in upstream dependencies (please report to those projects directly; we will track upstream advisories)
- Findings only achievable with attacker-controlled access to the local filesystem (this is the trust model — RAG runs locally on user machines)
- Denial-of-service via genuinely large input the user opted to index
- Issues already covered by the public dependency scanners (Snyk, Socket, CodeQL)

## Coordinated disclosure

We follow a **90-day coordinated disclosure** window from acknowledgement to public disclosure, extendable by mutual agreement if a fix needs more time. After a fix ships, we publish a GitHub Security Advisory with credit to the reporter (unless you request anonymity).

## Hall of fame

Reporters who help improve `knowledge-rag` security will be acknowledged in our [Security Advisories](https://github.com/lyonzin/knowledge-rag/security/advisories) and the project README, unless they prefer to remain anonymous.

## Bug bounty

This project does not currently offer a paid bug bounty. We deeply appreciate volunteer contributions to security and credit reporters publicly.

---

Thank you for helping keep `knowledge-rag` and its 70+ enterprise users safe.
