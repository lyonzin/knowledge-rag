# knowledge-rag Skills for AI Agents

> Drop-in skills that teach any MCP-aware AI agent — Claude Code, Codex, Cursor, Gemini CLI, GitHub Copilot, Windsurf, Cline, Zed — to use your local knowledge base as its primary source of truth. Install the ones you want, customize, ship.

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

### Option 1 — `knowledge-rag skills install` (every OS)

```bash
knowledge-rag skills install
```

This needs knowledge-rag itself, in a release newer than 5.0.0: `pip install knowledge-rag`, `pipx install knowledge-rag` or `uv tool install knowledge-rag`. With Node.js instead, `npx -y knowledge-rag skills install` runs the same command.

It looks for the AI coding assistants of the current user — a config folder in your profile or an executable on PATH, without changing anything — prints what it would do in each skills folder, and asks once per folder:

```text
Claude Code: C:\Users\you\.claude\skills
  10 not installed
Install the knowledge-rag skills for Claude Code? [Y/n]
```

Each skill goes in as a `<name>/SKILL.md` folder, the layout these assistants load:

| Assistant | Detected by | Skills folder | Official docs |
|---|---|---|---|
| Claude Code | `~/.claude` or `$CLAUDE_CONFIG_DIR`; `claude` | `~/.claude/skills`, or `$CLAUDE_CONFIG_DIR/skills` when set | [Skills](https://code.claude.com/docs/en/skills) |
| Codex CLI | `~/.codex` or `$CODEX_HOME`; `codex` | `~/.agents/skills` | [Skills](https://developers.openai.com/codex/skills) |
| GitHub Copilot CLI | `~/.copilot`; `copilot` | `~/.agents/skills` | [Agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills) |
| Gemini CLI | `~/.gemini`; `gemini` | `~/.agents/skills` | [Skills](https://geminicli.com/docs/cli/skills/) |
| Cursor | `~/.cursor`; `cursor` | `~/.agents/skills` | [Skills](https://cursor.com/docs/skills) |
| OpenCode | `~/.config/opencode`; `opencode` | `~/.agents/skills` | [Skills](https://opencode.ai/docs/skills/) |
| Windsurf | `~/.codeium/windsurf`; `windsurf` | `~/.agents/skills` | [Skills](https://docs.windsurf.com/windsurf/cascade/skills) |
| Zed | `~/.config/zed` or `%APPDATA%\Zed`; `zed` | `~/.agents/skills` | [Skills](https://zed.dev/docs/ai/skills) |
| Cline | `~/.cline`; `cline` | `~/.cline/skills` | [Skills](https://docs.cline.bot/customization/skills) |
| Aider | `~/.aider.conf.yml`; `aider` | none, see [Other assistants](#other-assistants) | [Conventions](https://aider.chat/docs/usage/conventions.html) |

`~` is your user profile: `%USERPROFILE%` on Windows (`C:\Users\you`, even when OneDrive redirects your Documents folder). The assistants that share `~/.agents/skills` get one question between them. Under WSL the installer also checks the Windows profile (`/mnt/c/Users/you`, for config folders only) and offers its folders marked "(Windows)".

Re-running is safe. A `.knowledge-rag-skills.json` file in each skills folder records the copies it installed: a re-run updates the ones you have not touched, keeps the ones you edited, and leaves symlinked skill folders alone; `uninstall` removes only unedited skill folders that hold nothing else. Start a new session in each assistant afterwards.

| Option | What it does |
|---|---|
| `install`, `uninstall`, `status` | The subcommand; `status` changes nothing |
| `-y`, `--yes` | Do not ask. When stdin is not a terminal (CI, a pipe), the command prints its plan and changes nothing without it |
| `--dry-run` | Print the plan and change nothing |
| `--harness NAME` | Only this assistant, detected or not (repeatable): `claude-code`, `codex`, `copilot`, `gemini`, `cursor`, `opencode`, `windsurf`, `zed`, `cline`, `aider` |
| `--target DIR` | Only this skills folder (repeatable) |
| `--project` | Only `./.claude/skills` in the current folder, to commit the skills with a project |
| `--skill NAME` | Only these skills, comma-separated or repeated; `--only` is the same option |

```bash
knowledge-rag skills status
knowledge-rag skills install --dry-run
knowledge-rag skills install --harness claude-code --yes
knowledge-rag skills install --skill rag-check-first,rag-cite-sources,rag-onboard-context
knowledge-rag skills uninstall
```

Cursor, OpenCode and VS Code's local Copilot sessions read `~/.claude/skills` as well as `~/.agents/skills`, so with both folders installed they can list each skill twice. Answer no to one of the two questions if that bothers you.

### Option 2 — one-line wrappers

Linux, macOS, WSL and Git Bash:

```bash
curl -fsSL https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.sh | sh
```

Windows PowerShell 5.1 or PowerShell 7:

```powershell
irm https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.ps1 | iex
```

Both run `knowledge-rag skills install` and forward any option above: `sh skills/install.sh status` or `curl ... | sh -s -- --dry-run`; from a checkout on Windows, `powershell -ExecutionPolicy Bypass -File skills\install.ps1 --dry-run`. When knowledge-rag is missing they say how to install it (exit code 127). When it is 5.0.0 or older they ask you to upgrade (exit code 1) instead of running it, because those releases ignore `skills` and would start the MCP server. If Git Bash's default mintty window answers that stdin is not a terminal, add `--yes` or run from Windows Terminal.

### Option 3 — `skills.sh` (Node)

```bash
npx skills add lyonzin/knowledge-rag
```

The [Vercel Labs `skills` CLI](https://github.com/vercel-labs/skills) discovers this repo's `skills/**/SKILL.md` layout and asks which assistants to install for. It installs into the current project by default; add `-g` for all of your projects, and `--skill NAME` (repeated) for a subset.

### Manual install

The repo groups the skills by category (`foundation/`, `workflow/`, `maintenance/`, `domain/`); an assistant wants the skill folders side by side in its skills folder from the table above:

```bash
git clone https://github.com/lyonzin/knowledge-rag.git
mkdir -p ~/.claude/skills
cp -R knowledge-rag/skills/*/rag-* ~/.claude/skills/
```

That gives `~/.claude/skills/rag-check-first/SKILL.md` and so on. The skills are auto-discovered by their frontmatter `description` — trigger them by typing keywords that match, or invoke one explicitly, e.g. `/rag-check-first` in Claude Code.

The `install.sh` of earlier versions copied the skills as `~/.claude/skills/rag-*.md` files, which Claude Code never loaded because it needs `<name>/SKILL.md`. Delete those files.

### Other assistants

Aider and other clients without a skills folder still use the MCP tools; give them the skill text instead:

1. **Read it into the session** — `aider --read path/to/rag-check-first/SKILL.md`, or list the files under `read:` in `.aider.conf.yml`. Aider takes the frontmatter as plain text.
2. **System prompt or rules file** — paste the skill body into the client's global instructions, or into an `AGENT_RULES.md` that the system prompt tells it to follow.

Every skill in this folder works in either mode — the frontmatter is metadata only, the body is pure instructions.

---

## Quick tour — pick your first 3

New to skills? Start with these three. They are the highest-ROI baseline for any AI agent talking to knowledge-rag:

| Skill | What it does | When it kicks in |
|---|---|---|
| [`rag-check-first`](foundation/rag-check-first/SKILL.md) | Forces a `search_knowledge` call before answering any technical claim | Every code / architecture / security / research question |
| [`rag-cite-sources`](foundation/rag-cite-sources/SKILL.md) | Every claim cites the file it came from; a line only once checked in `get_document` | Every response that references RAG data |
| [`rag-onboard-context`](foundation/rag-onboard-context/SKILL.md) | First interaction of a new session: reads `get_index_stats` and probes a topic or two | Session start |

Add more from the [full catalog](CATALOG.md) as your workflow grows.

---

## How the skills are organized

Every skill is a `<category>/<name>/SKILL.md` file with the same structure:

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

`knowledge-rag skills install` keeps a skill you edited: it replaces only copies that still match what it installed.

---

## Why not just put this in the system prompt?

You can. But:

- **Skills are versionable** — commit them to the repo, review changes in PRs, revert if a rule backfires
- **Skills compose** — pick 3 for one project, 8 for another, without duplicating body text
- **Skills auto-trigger** — Claude Code matches `description` fields against the user's message; you do not have to remember to invoke them
- **Skills are shareable** — a team member joining an existing project inherits the exact behavioural discipline the previous author installed

---

## Contributing new skills

Have a workflow that plays well with knowledge-rag? Open a PR — one `<category>/rag-<name>/SKILL.md` folder per skill, follow the template above, add an entry to [CATALOG.md](CATALOG.md).

**Guidelines:**
- Name in `kebab-case`, prefixed with `rag-` to make the origin clear; the folder name and the frontmatter `name` must match
- `description` must be specific enough to auto-trigger correctly (avoid "helps with X" — say "when the user asks Y, do Z")
- Every step must map to a concrete MCP tool call or explicit reasoning step
- Include ≥1 real example and ≥1 edge case
- A new category folder also needs a `force-include` entry in `pyproject.toml`, so the wheel ships it

---

## License

Same as the parent project: [MIT](../LICENSE). Fork, modify, distribute — the license does not care.
