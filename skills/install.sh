#!/usr/bin/env bash
# knowledge-rag skills installer — Linux / macOS / WSL / Git Bash on Windows.
#
# Copies every rag-*.md skill into your Claude Code skills directory.
# Zero external dependencies, safe re-runs, clear output.
#
# Usage:
#   # One-liner from the internet (recommended for most users):
#   curl -fsSL https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.sh | bash
#
#   # Or after cloning the repo:
#   bash skills/install.sh                    # install globally to ~/.claude/skills
#   bash skills/install.sh --project          # install into ./.claude/skills (project-scoped)
#   bash skills/install.sh --target <path>    # install into a custom directory
#   bash skills/install.sh --only rag-check-first,rag-cite-sources   # subset
#   bash skills/install.sh --dry-run          # show what would happen, do nothing
#   bash skills/install.sh --help
#
# Exit codes: 0 ok · 1 wrong invocation · 2 nothing to copy · 3 destination unwritable

set -euo pipefail

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

REPO_RAW="https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills"

# Skill → category mapping (matches the folder layout)
declare -A SKILL_CATEGORY=(
  ["rag-check-first"]="foundation"
  ["rag-cite-sources"]="foundation"
  ["rag-onboard-context"]="foundation"
  ["rag-deep-dive"]="workflow"
  ["rag-web-fallback"]="workflow"
  ["rag-troubleshoot"]="workflow"
  ["rag-code-review"]="workflow"
  ["rag-index-decisions"]="maintenance"
  ["rag-evaluate-quality"]="maintenance"
  ["rag-security-first"]="domain"
)
REPO_TARBALL="https://github.com/lyonzin/knowledge-rag/archive/refs/heads/master.tar.gz"

SKILLS=(
  "rag-check-first"
  "rag-cite-sources"
  "rag-onboard-context"
  "rag-deep-dive"
  "rag-web-fallback"
  "rag-troubleshoot"
  "rag-code-review"
  "rag-index-decisions"
  "rag-security-first"
  "rag-evaluate-quality"
)

# ----------------------------------------------------------------------------
# ANSI colors (fall back to plain when stdout is not a terminal)
# ----------------------------------------------------------------------------

if [ -t 1 ]; then
  BOLD=$(printf '\033[1m')
  DIM=$(printf '\033[2m')
  RED=$(printf '\033[31m')
  GREEN=$(printf '\033[32m')
  YELLOW=$(printf '\033[33m')
  BLUE=$(printf '\033[34m')
  RESET=$(printf '\033[0m')
else
  BOLD="" DIM="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

log()      { printf '%s%s%s\n' "$BLUE"   "$*" "$RESET"; }
ok()       { printf '%s✓ %s%s\n' "$GREEN"  "$*" "$RESET"; }
warn()     { printf '%s! %s%s\n' "$YELLOW" "$*" "$RESET"; }
err()      { printf '%s✗ %s%s\n' "$RED"    "$*" "$RESET" >&2; }
title()    { printf '\n%s%s%s\n\n' "$BOLD" "$*" "$RESET"; }

usage() {
  cat <<EOF
${BOLD}knowledge-rag skills installer${RESET}

Copies the 10 rag-*.md skills into your Claude Code skills directory.

${BOLD}Usage:${RESET}
  bash skills/install.sh [options]
  curl -fsSL https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.sh | bash

${BOLD}Options:${RESET}
  --project              Install to ./.claude/skills (project-scoped)
  --target <path>        Install to a custom directory
  --only <a,b,c>         Install only the listed skills (comma-separated, without .md)
  --dry-run              Show what would happen, do not copy anything
  --list                 List available skills and exit
  --help                 Show this help

${BOLD}Default:${RESET} installs all 10 skills to ~/.claude/skills (global, all projects benefit)

${BOLD}Examples:${RESET}
  bash skills/install.sh
  bash skills/install.sh --project
  bash skills/install.sh --only rag-check-first,rag-cite-sources,rag-onboard-context
  bash skills/install.sh --target ~/my-agent-skills

${BOLD}After install:${RESET} restart Claude Code. Skills auto-discover via their frontmatter.

${BOLD}Docs:${RESET} https://github.com/lyonzin/knowledge-rag/tree/master/skills
EOF
}

list_skills() {
  echo "Available skills:"
  for s in "${SKILLS[@]}"; do
    echo "  - $s"
  done
}

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------

TARGET=""
MODE="global"
ONLY=""
DRY_RUN=false

while [ $# -gt 0 ]; do
  case "$1" in
    --project)   MODE="project"; shift ;;
    --target)    TARGET="${2:-}"; MODE="custom"; shift 2 ;;
    --only)      ONLY="${2:-}"; shift 2 ;;
    --dry-run)   DRY_RUN=true; shift ;;
    --list)      list_skills; exit 0 ;;
    --help|-h)   usage; exit 0 ;;
    *)
      err "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

case "$MODE" in
  global)   TARGET="${TARGET:-$HOME/.claude/skills}" ;;
  project)  TARGET="$(pwd)/.claude/skills" ;;
  custom)   [ -n "$TARGET" ] || { err "--target requires a path"; exit 1; } ;;
esac

# Expand ~ safely if user passed literal tilde in --target
TARGET="${TARGET/#\~/$HOME}"

# ----------------------------------------------------------------------------
# Filter skills by --only
# ----------------------------------------------------------------------------

TO_INSTALL=()
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a REQUESTED <<< "$ONLY"
  for req in "${REQUESTED[@]}"; do
    req_trim="$(echo "$req" | xargs)"
    matched=false
    for available in "${SKILLS[@]}"; do
      if [ "$available" = "$req_trim" ]; then
        TO_INSTALL+=("$req_trim"); matched=true; break
      fi
    done
    if ! $matched; then
      warn "Skipping unknown skill: $req_trim (use --list to see available)"
    fi
  done
  if [ "${#TO_INSTALL[@]}" -eq 0 ]; then
    err "No valid skills selected."
    exit 2
  fi
else
  TO_INSTALL=("${SKILLS[@]}")
fi

# ----------------------------------------------------------------------------
# Discover source: local checkout OR remote download
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SOURCE_MODE=""
TMP_DIR=""

if [ -f "$SCRIPT_DIR/foundation/rag-check-first/SKILL.md" ]; then
  SOURCE_MODE="local"
  SOURCE_DIR="$SCRIPT_DIR"
else
  # Remote install path: pipe from curl.
  SOURCE_MODE="remote"
  if command -v curl >/dev/null 2>&1; then
    :
  elif command -v wget >/dev/null 2>&1; then
    :
  else
    err "Neither curl nor wget is available — install one or clone the repo manually."
    exit 1
  fi
fi

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------

title "knowledge-rag — skills installer"

echo "  Mode:         $MODE"
echo "  Target:       $TARGET"
echo "  Source:       $SOURCE_MODE"
echo "  Skills:       ${#TO_INSTALL[@]} of ${#SKILLS[@]}"
$DRY_RUN && echo "  Dry-run:      yes (no files will be written)"
echo

# ----------------------------------------------------------------------------
# Ensure target exists (unless dry-run)
# ----------------------------------------------------------------------------

if ! $DRY_RUN; then
  if ! mkdir -p "$TARGET" 2>/dev/null; then
    err "Could not create target directory: $TARGET"
    exit 3
  fi
  if [ ! -w "$TARGET" ]; then
    err "Target directory is not writable: $TARGET"
    exit 3
  fi
fi

# ----------------------------------------------------------------------------
# Copy or download each skill
# ----------------------------------------------------------------------------

COPIED=0
SKIPPED=0

for skill in "${TO_INSTALL[@]}"; do
  dest="$TARGET/${skill}.md"

  if $DRY_RUN; then
    if [ -f "$dest" ]; then
      log "[DRY] would overwrite $dest"
    else
      log "[DRY] would install  $dest"
    fi
    COPIED=$((COPIED+1))
    continue
  fi

  cat="${SKILL_CATEGORY[$skill]}"

  if [ "$SOURCE_MODE" = "local" ]; then
    src="$SOURCE_DIR/${cat}/${skill}/SKILL.md"
    if [ ! -f "$src" ]; then
      warn "Source missing: $src — skipping"
      SKIPPED=$((SKIPPED+1))
      continue
    fi
    cp "$src" "$dest"
    ok "installed $skill.md"
    COPIED=$((COPIED+1))
  else
    url="$REPO_RAW/${cat}/${skill}/SKILL.md"
    if command -v curl >/dev/null 2>&1; then
      if curl -fsSL --retry 3 "$url" -o "$dest"; then
        ok "downloaded $skill.md"
        COPIED=$((COPIED+1))
      else
        warn "download failed: $url — skipping"
        SKIPPED=$((SKIPPED+1))
      fi
    else
      if wget -q -O "$dest" "$url"; then
        ok "downloaded $skill.md"
        COPIED=$((COPIED+1))
      else
        warn "download failed: $url — skipping"
        SKIPPED=$((SKIPPED+1))
      fi
    fi
  fi
done

# ----------------------------------------------------------------------------
# Final report
# ----------------------------------------------------------------------------

echo
if $DRY_RUN; then
  title "Dry-run complete"
  log "$COPIED skills would be installed to $TARGET"
elif [ "$COPIED" -gt 0 ]; then
  title "Install complete"
  ok "$COPIED skills installed to $TARGET"
  [ "$SKIPPED" -gt 0 ] && warn "$SKIPPED skipped (see warnings above)"

  cat <<EOF

${BOLD}Next steps:${RESET}
  1. Restart your MCP client (Claude Code, Cursor, Windsurf, ...)
  2. Skills auto-discover via their frontmatter description
  3. Trigger by keyword in your prompt, or invoke explicitly with /rag-check-first

${BOLD}Docs:${RESET}
  https://github.com/lyonzin/knowledge-rag/tree/master/skills
  https://github.com/lyonzin/knowledge-rag/blob/master/skills/CATALOG.md

${BOLD}Uninstall (if ever needed):${RESET}
  rm -f "$TARGET"/rag-*.md
EOF
else
  err "Nothing was installed."
  exit 2
fi
