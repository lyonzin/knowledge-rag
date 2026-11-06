# Skills Catalog

10 skills covering the full workflow of an AI agent backed by [knowledge-rag](../README.md).

**Install all of them for every AI coding assistant on your machine with:**

```bash
knowledge-rag skills install
```

The repo layout also works with [skills.sh](https://www.skills.sh/): `npx skills add lyonzin/knowledge-rag`.

**Legend:**
- 🎯 **Foundation** — the 3 skills you should install first
- 🧠 **Workflow** — chained multi-tool patterns
- 🏢 **Domain** — specialized for a preset (security, dev, research)
- 🔁 **Maintenance** — keep the index healthy

---

| # | Skill | Kind | One-liner | Depends on MCP tools |
|---|---|:---:|---|---|
| 1 | [`rag-check-first`](foundation/rag-check-first/SKILL.md) | 🎯 | Search the corpus **before** answering any technical claim | `search_knowledge` |
| 2 | [`rag-cite-sources`](foundation/rag-cite-sources/SKILL.md) | 🎯 | Every technical claim ships with its source file; a line only once checked in `get_document` | `search_knowledge`, `get_document` |
| 3 | [`rag-onboard-context`](foundation/rag-onboard-context/SKILL.md) | 🎯 | First interaction of a session probes what is indexed | `get_index_stats`, `search_knowledge`, `list_documents` |
| 4 | [`rag-deep-dive`](workflow/rag-deep-dive/SKILL.md) | 🧠 | 3-step drill: search → fetch → find-similar | `search_knowledge`, `get_document`, `search_similar` |
| 5 | [`rag-web-fallback`](workflow/rag-web-fallback/SKILL.md) | 🧠 | Only hit the web when local RAG comes back empty | `search_knowledge`, then external `WebSearch` |
| 6 | [`rag-troubleshoot`](workflow/rag-troubleshoot/SKILL.md) | 🧠 | Bug / error / stack trace → RAG first for prior fixes | `search_knowledge` (with error signature) |
| 7 | [`rag-code-review`](workflow/rag-code-review/SKILL.md) | 🧠 | Code review consults ADRs / patterns before commenting | `search_knowledge`, `search_similar` |
| 8 | [`rag-index-decisions`](maintenance/rag-index-decisions/SKILL.md) | 🔁 | After making an architectural decision, index it back | `add_document`, `add_from_url` |
| 9 | [`rag-security-first`](domain/rag-security-first/SKILL.md) | 🏢 | Security tasks: MITRE / CVE / threat context first | `search_knowledge` (cybersecurity preset) |
| 10 | [`rag-evaluate-quality`](maintenance/rag-evaluate-quality/SKILL.md) | 🔁 | Periodically measure retrieval quality (MRR@5 / Recall@5) | `evaluate_retrieval`, `get_index_stats` |

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
`rag-evaluate-quality` → (if quality drops) `rag-index-decisions` on new material → `reindex_documents()` if files changed outside the tools

`add_document`, `update_document` and `add_from_url` index as they write, and `reindex_documents()` picks up new, changed and deleted files. `reindex_documents(force=True)` re-embeds every file, so keep it for a change of query/passage prefixes or chunking settings.

---

## Installation

Pick the one you like; [README.md#installation](README.md#installation) has the details:

### 1. `knowledge-rag skills install` — every OS

```bash
knowledge-rag skills install          # after pip install knowledge-rag (a release newer than 5.0.0)
npx -y knowledge-rag skills install   # the same through npm
```

It finds the AI coding assistants of the current user and asks once per skills folder before copying `<name>/SKILL.md` folders into it.

### 2. One-line wrappers

```bash
curl -fsSL https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.sh | sh
```

```powershell
irm https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.ps1 | iex
```

Both run `knowledge-rag skills install` and say what to install when knowledge-rag is missing or too old.

### 3. Via `skills.sh` (Node)

```bash
npx skills add lyonzin/knowledge-rag       # this project; add -g for all of yours
```

The [Vercel Labs skills CLI](https://github.com/vercel-labs/skills) discovers this repo's `skills/**/SKILL.md` layout automatically and asks which assistants to install for.

### 4. Manual copy

See [README.md#manual-install](README.md#manual-install).

---

## Choosing skills for your project

**Minimal setup (3 skills):** `rag-check-first` + `rag-cite-sources` + `rag-onboard-context`. Covers 80% of the value.

**Standard setup (6 skills):** minimal + `rag-deep-dive` + `rag-web-fallback` + `rag-troubleshoot`. Full daily-work coverage.

**Enterprise setup (all 10):** everything above + `rag-code-review` + `rag-index-decisions` + `rag-security-first` + `rag-evaluate-quality`. Adds review discipline, feedback loop, security emphasis, and quality monitoring.

**Domain-specific (pick one):**
- Security team → `rag-security-first` is mandatory
- Research lab → `rag-cite-sources` + `rag-deep-dive` are mandatory
- Enterprise SRE → `rag-evaluate-quality` + `rag-index-decisions` are mandatory

Install a subset with either tool:

```bash
# knowledge-rag (the wrappers take the same option)
knowledge-rag skills install --skill rag-check-first,rag-cite-sources,rag-onboard-context

# skills.sh: repeat --skill (or -s) once per skill
npx skills add lyonzin/knowledge-rag --skill rag-check-first --skill rag-cite-sources --skill rag-onboard-context
```

---

## Contributing

New skill? Open a PR:

1. Pick a category folder: `skills/foundation/`, `skills/workflow/`, `skills/maintenance/`, `skills/domain/`
2. Create `<category>/rag-your-skill/SKILL.md` following the template in an existing skill
3. Add a row to this catalog with kind + one-liner + MCP tools used
4. Cross-link from at least one existing skill's "Related skills" section
5. Include at least 1 example query + 1 edge case

See [README.md#contributing-new-skills](README.md#contributing-new-skills) for full guidelines.
