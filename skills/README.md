# knowledge-rag Skills for AI Agents

> Drop-in skills that teach any MCP-aware AI agent — Claude Code, Cursor, Windsurf, Cline, Zed, VS Code Copilot — to use your local knowledge base as its primary source of truth. Copy the ones you want, customize, ship.

**What these skills do:** they instruct the AI to reach for the [13 knowledge-rag MCP tools](../docs/API.md) at the right moments — before writing code, before answering claims, before searching the web — so the agent leans on YOUR indexed corpus (your docs, your ADRs, your runbooks, your MITRE mapping, your team's context) instead of guessing from training data.

**Related:**
- [Full skills catalog →](CATALOG.md)
- [API reference for the 13 MCP tools →](../docs/API.md)
- [Main README →](../README.md)

---

## Why skills matter for a RAG-backed agent

Installing knowledge-rag gives your AI 13 tools. It does not tell the AI **when** to use them. Left on its own, most models will:

- Answer from training data even when your indexed corpus has a more recent / more specific answer
- Skip citation because "the user did not ask for it"
- Search the web before checking local knowledge
- Forget to index important decisions after making them

**Skills close the loop.** They are the discipline layer that turns "AI with access to RAG" into "AI that actually uses RAG first."

---

## Installation

### Option 1 — Via `skills.sh` (the shortest path)

```bash
npx skills add lyonzin/knowledge-rag
```

This uses the [Vercel Labs `skills` CLI](https://github.com/vercel-labs/skills) which auto-discovers this repo's `skills/**/SKILL.md` layout and installs into `.claude/skills/`. Zero clone, zero curl, one command.

### Option 2 — Via our `install.sh` (no Node required)

```bash
curl -fsSL https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.sh | bash
```

Works on **Linux, macOS, WSL, and Git Bash on Windows**. Downloads and installs all 10 skills to `~/.claude/skills/`, skips duplicates gracefully, and prints a summary. Restart Claude Code — skills auto-discover via their frontmatter.

> **Windows users without WSL or Git Bash:** either use Option 1 (needs Node.js) or fall back to the [manual install](#manual-install-claude-code) below.

### Installer options (`install.sh`)

| Option | What it does |
|---|---|
| `--project` | Install to `./.claude/skills` (project-scoped, not global) |
| `--target <path>` | Install to a custom directory |
| `--only rag-check-first,rag-cite-sources` | Install a subset only |
| `--dry-run` | Show what would happen, do not write files |
| `--list` | List available skills |
| `--help` | Show the full help |

Examples:

```bash
# Install only the 3 foundation skills, project-scoped
bash skills/install.sh --project --only rag-check-first,rag-cite-sources,rag-onboard-context

# Dry-run to preview what would land where
bash skills/install.sh --dry-run

# Install to a custom folder
bash skills/install.sh --target ~/my-agent-skills
```

### Manual install (Claude Code)

If you prefer to see the files before installing:

```bash
git clone https://github.com/lyonzin/knowledge-rag.git

# Global — all projects benefit
mkdir -p ~/.claude/skills
# The repo organizes skills by category (foundation/workflow/maintenance/domain);
# copy the SKILL.md files out and rename them per skill:
for f in knowledge-rag/skills/*/*/SKILL.md; do
  skill=$(basename "$(dirname "$f")")
  cp "$f" ~/.claude/skills/"$skill".md
done
```

Restart Claude Code. The skills are auto-discovered by their frontmatter `description` — trigger by typing keywords that match, or invoke explicitly with `/rag-check-first`, `/rag-cite-sources`, etc.

### Cursor

Cursor supports custom rules and `.cursor/rules/*.mdc` files. Adapt each skill:

```bash
mkdir -p .cursor/rules
for f in knowledge-rag/skills/rag-*.md; do
  name=$(basename "$f" .md)
  cp "$f" ".cursor/rules/${name}.mdc"
done
```

Skills apply based on the rule's frontmatter `alwaysApply` / `globs` fields. Adjust per your project.

### Windsurf, Cline, Zed, VS Code Copilot

These clients consume the MCP tools directly without a native "skill" concept. You have two options:

1. **System prompt injection** — paste the skill body into your global system prompt / rules file (usually `.windsurf/rules.md`, `.clinerules`, `.zed/prompt.md`, `.github/copilot-instructions.md`).
2. **Reference from a project doc** — add a `AGENT_RULES.md` to your repo and reference it in the client's system prompt: "Follow the rules in `AGENT_RULES.md` for every task."

Every skill in this folder is written to work in either mode — the frontmatter is metadata only, the body is pure instructions.

---

## Quick tour — pick your first 3

New to skills? Start with these three. They are the highest-ROI baseline for any AI agent talking to knowledge-rag:

| Skill | What it does | When it kicks in |
|---|---|---|
| [`rag-check-first`](rag-check-first.md) | Forces a `search_knowledge` call before answering any technical claim | Every code / architecture / security / research question |
| [`rag-cite-sources`](rag-cite-sources.md) | Every claim carries a `path:line` citation from the corpus | Every response that references RAG data |
| [`rag-onboard-context`](rag-onboard-context.md) | First interaction of a new session: probes `get_index_stats` + `list_categories` | Session start |

Add more from the [full catalog](CATALOG.md) as your workflow grows.

---

## How the skills are organized

Every file follows the same structure:

```markdown
---
name: kebab-case-slug
description: One-line trigger sentence — Claude Code uses this to auto-invoke
metadata:
  type: rag-workflow
  target: any-mcp-client
---

## When to use this skill
Concrete triggers — what the user asked, what the code looks like, what phase you are in.

## What this skill does
The invariant: what the AI is committing to do differently.

## Steps
Numbered protocol with the exact MCP tools to call.

## Examples
Real-world queries + expected tool call sequences.

## Edge cases
When to bail out or fall back.

## Related skills
Cross-links to skills that chain naturally.
```

You can read a skill in 60 seconds. You can adapt one to your project in 5 minutes.

---

## Customizing skills for your project

Skills are just markdown. **Edit them.** Real-world patterns:

- **Change the trigger keywords** in the `description` field to match your team's vocabulary
- **Add project-specific queries** in the `Examples` section (e.g. "search for our internal auth service ADRs")
- **Chain to your own tools** — if your agent has more MCP servers, add "then call `your_tool()` with the results"
- **Tighten the fallback rules** — enterprise deployments often want `rag-web-fallback` to hard-block web searches entirely

---

## Why not just put this in the system prompt?

You can. But:

- **Skills are versionable** — commit them to the repo, review changes in PRs, revert if a rule backfires
- **Skills compose** — pick 3 for one project, 8 for another, without duplicating body text
- **Skills auto-trigger** — Claude Code matches `description` fields against the user's message; you do not have to remember to invoke them
- **Skills are shareable** — a team member joining an existing project inherits the exact behavioural discipline the previous author installed

---

## Contributing new skills

Have a workflow that plays well with knowledge-rag? Open a PR — one file per skill, follow the template above, add an entry to [CATALOG.md](CATALOG.md).

**Guidelines:**
- Name in `kebab-case`, prefixed with `rag-` to make the origin clear
- `description` must be specific enough to auto-trigger correctly (avoid "helps with X" — say "when the user asks Y, do Z")
- Every step must map to a concrete MCP tool call or explicit reasoning step
- Include ≥1 real example and ≥1 edge case

---

## License

Same as the parent project: [MIT](../LICENSE). Fork, modify, distribute — the license does not care.
