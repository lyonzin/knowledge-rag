# Skills Catalog

10 skills covering the full workflow of an AI agent backed by [knowledge-rag](../README.md).

**Legend:**
- 🎯 **Foundation** — the 3 skills you should install first
- 🧠 **Workflow** — chained multi-tool patterns
- 🏢 **Domain** — specialized for a preset (security, dev, research)
- 🔁 **Maintenance** — keep the index healthy

---

| # | Skill | Kind | One-liner | Depends on MCP tools |
|---|---|:---:|---|---|
| 1 | [`rag-check-first`](rag-check-first.md) | 🎯 | Search the corpus **before** answering any technical claim | `search_knowledge` |
| 2 | [`rag-cite-sources`](rag-cite-sources.md) | 🎯 | Every technical claim ships with `path:line` citations | `search_knowledge`, `get_document` |
| 3 | [`rag-onboard-context`](rag-onboard-context.md) | 🎯 | First interaction of a session probes what is indexed | `get_index_stats`, `list_categories`, `list_documents` |
| 4 | [`rag-deep-dive`](rag-deep-dive.md) | 🧠 | 3-step drill: search → fetch → find-similar | `search_knowledge`, `get_document`, `search_similar` |
| 5 | [`rag-web-fallback`](rag-web-fallback.md) | 🧠 | Only hit the web when local RAG comes back empty | `search_knowledge`, then external `WebSearch` |
| 6 | [`rag-troubleshoot`](rag-troubleshoot.md) | 🧠 | Bug / error / stack trace → RAG first for prior fixes | `search_knowledge` (with error signature) |
| 7 | [`rag-code-review`](rag-code-review.md) | 🧠 | Code review consults ADRs / patterns before commenting | `search_knowledge`, `search_similar` |
| 8 | [`rag-index-decisions`](rag-index-decisions.md) | 🔁 | After making an architectural decision, index it back | `add_document`, `add_from_url` |
| 9 | [`rag-security-first`](rag-security-first.md) | 🏢 | Security tasks: MITRE / CVE / threat context first | `search_knowledge` (cybersecurity preset) |
| 10 | [`rag-evaluate-quality`](rag-evaluate-quality.md) | 🔁 | Periodically measure retrieval quality (MRR / Recall / Precision) | `evaluate_retrieval`, `get_index_stats` |

---

## Skill chains (recommended combos)

Skills work well solo. They work **great** when chained. Common patterns:

### Chain: "Answering a design question"
`rag-check-first` → `rag-deep-dive` → `rag-cite-sources`

The agent searches for context, drills into the top matches, then answers with citations.

### Chain: "Fixing a bug reported in Slack"
`rag-check-first` → `rag-troubleshoot` → `rag-cite-sources` → `rag-index-decisions` (if the fix is worth remembering)

Search RAG first for prior fixes, apply, cite the source, then index the new decision for the next occurrence.

### Chain: "Code review PR opened"
`rag-onboard-context` (fresh session) → `rag-code-review` → `rag-cite-sources`

Understand what the project cares about, review against those standards, cite the ADR being enforced.

### Chain: "Security incident triage"
`rag-onboard-context` → `rag-security-first` → `rag-deep-dive` → `rag-cite-sources`

Load session context, prioritize the security preset, drill down, cite the MITRE technique / runbook.

### Chain: "Weekly maintenance"
`rag-evaluate-quality` → (if quality drops) `rag-index-decisions` on new material → `reindex_documents(force=True)`

---

## Installation

See [README.md#installation](README.md#installation) for per-client installation. Short version:

```bash
mkdir -p ~/.claude/skills
cp skills/rag-*.md ~/.claude/skills/
```

Restart Claude Code. All 10 skills auto-discover.

---

## Choosing skills for your project

**Minimal setup (3 skills):** `rag-check-first` + `rag-cite-sources` + `rag-onboard-context`. Covers 80% of the value.

**Standard setup (6 skills):** minimal + `rag-deep-dive` + `rag-web-fallback` + `rag-troubleshoot`. Full daily-work coverage.

**Enterprise setup (all 10):** everything above + `rag-code-review` + `rag-index-decisions` + `rag-security-first` + `rag-evaluate-quality`. Adds review discipline, feedback loop, security emphasis, and quality monitoring.

**Domain-specific (pick one):**
- Security team → `rag-security-first` is mandatory
- Research lab → `rag-cite-sources` + `rag-deep-dive` are mandatory
- Enterprise SRE → `rag-evaluate-quality` + `rag-index-decisions` are mandatory

---

## Contributing

New skill? Open a PR:

1. Create `skills/rag-your-skill.md` following the template in an existing skill
2. Add a row to this catalog with kind + one-liner + MCP tools used
3. Cross-link from at least one existing skill's "Related skills" section
4. Include at least 1 example query + 1 edge case

See [README.md#contributing-new-skills](README.md#contributing-new-skills) for full guidelines.
