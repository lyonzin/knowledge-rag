#!/bin/sh
# knowledge-rag skills installer for Linux, macOS, WSL and Git Bash on Windows.
#
# A thin wrapper around `knowledge-rag skills`, which detects your AI coding
# assistants and asks before installing into each one's skills folder:
#
#   curl -fsSL https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.sh | sh
#   sh skills/install.sh                             # install, the default subcommand
#   sh skills/install.sh status                      # or uninstall
#   sh skills/install.sh --dry-run --harness claude-code
#
# Every option goes to `knowledge-rag skills`; see `knowledge-rag skills install --help`.
# Exit codes are those of `knowledge-rag skills`, 127 when knowledge-rag is not
# installed, and 1 when it is too old to have the skills installer.
set -eu

case "${1-}" in
    install | uninstall | status | -h | --help) ;;
    *) set -- install "$@" ;;
esac

# knowledge-rag-skills runs `knowledge-rag skills`. Releases up to 5.0.0 lack it, and their
# `knowledge-rag` would ignore `skills` and start the MCP server, so never call that one.
if ! command -v knowledge-rag-skills >/dev/null 2>&1; then
    if command -v knowledge-rag >/dev/null 2>&1; then
        printf '%s\n' \
            "This knowledge-rag is too old for the skills installer. Upgrade it, then run this again:" \
            "  pip install -U knowledge-rag    (or: pipx upgrade knowledge-rag, uv tool upgrade knowledge-rag)" >&2
        exit 1
    fi
    printf '%s\n' \
        "knowledge-rag is not installed, or not on PATH. Install it, then run this again:" \
        "  pip install knowledge-rag    (or: pipx install knowledge-rag, uv tool install knowledge-rag)" \
        "Or run the same installer through npm, which needs Node.js and Python 3.11 or later:" \
        "  npx -y knowledge-rag skills install" \
        "Installed it in a virtual environment? Activate that environment first." >&2
    exit 127
fi

# Under `curl ... | sh` stdin is this script, so the answers come from the terminal.
if [ ! -t 0 ] && (exec </dev/tty) 2>/dev/null; then
    exec knowledge-rag-skills "$@" </dev/tty
fi
exec knowledge-rag-skills "$@"
