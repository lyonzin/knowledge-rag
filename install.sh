#!/usr/bin/env bash
# ╔═══════════════════════════════════════════════════════════════════╗
# ║                                                                   ║
# ║              KNOWLEDGE RAG — INSTALLER v3.0 (bash)                ║
# ║        Cross-platform, multi-LLM-client — Linux + macOS           ║
# ║                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝
#
# Thin wrapper that finds a supported Python (3.11 or 3.12) and delegates
# every installation step to install.py. See install.py --help for flags.
#
# Autor:   Ailton Rocha (Lyon.)
# Versão:  3.0.0
# Data:    2026-07-02

set -euo pipefail

# ─── Guard: must be bash, not /bin/sh ──────────────────────────────────────
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This installer requires bash. Run it with: bash install.sh $*" >&2
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_PY="$SCRIPT_DIR/install.py"

if [ ! -f "$INSTALL_PY" ]; then
    echo "[-] install.py not found next to install.sh (looked at $INSTALL_PY)" >&2
    exit 1
fi

# ─── Colors (auto-disabled when not a TTY / NO_COLOR set) ─────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_CYAN="\033[36m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"
    C_RED="\033[31m"; C_GRAY="\033[90m"; C_RESET="\033[0m"
else
    C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_GRAY=""; C_RESET=""
fi

info() { printf "%b[*]%b %s\n" "$C_CYAN" "$C_RESET" "$1"; }
ok()   { printf "%b[+]%b %s\n" "$C_GREEN" "$C_RESET" "$1"; }
warn() { printf "%b[!]%b %s\n" "$C_YELLOW" "$C_RESET" "$1"; }
err()  { printf "%b[-]%b %s\n" "$C_RED" "$C_RESET" "$1" >&2; }

# ─── Python detection (mirror of install.py fallback list) ────────────────
find_python() {
    local uname_s
    uname_s="$(uname -s 2>/dev/null || echo unknown)"

    local candidates=("python3.12" "python3.11")

    if [ "$uname_s" = "Linux" ]; then
        candidates+=(
            "/usr/bin/python3.12" "/usr/bin/python3.11"
            "/usr/local/bin/python3.12" "/usr/local/bin/python3.11"
        )
    elif [ "$uname_s" = "Darwin" ]; then
        candidates+=(
            "/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3.11"
            "/usr/local/opt/python@3.12/bin/python3.12"
            "/usr/local/opt/python@3.11/bin/python3.11"
            "/usr/local/bin/python3.12" "/usr/local/bin/python3.11"
        )
    fi
    candidates+=("python3" "python")

    local cmd version
    for cmd in "${candidates[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1 && [ ! -x "$cmd" ]; then
            continue
        fi
        version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        if [ "$version" = "3.11" ] || [ "$version" = "3.12" ]; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

print_python_install_hints() {
    echo ""
    warn "No supported Python (3.11 or 3.12) found. Install one first:"
    if command -v apt >/dev/null 2>&1; then
        echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev"
    fi
    if command -v dnf >/dev/null 2>&1; then
        echo "  Fedora/RHEL:   sudo dnf install python3.12 python3.12-devel"
    fi
    if command -v pacman >/dev/null 2>&1; then
        echo "  Arch:          sudo pacman -S python"
    fi
    if command -v apk >/dev/null 2>&1; then
        echo "  Alpine:        sudo apk add python3 py3-pip"
    fi
    if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
        echo "  macOS:         brew install python@3.12"
    fi
    echo "  Any platform:  pyenv install 3.12 && pyenv global 3.12"
    printf "%bNOTE: Python 3.13+ is NOT supported (onnxruntime).%b\n" "$C_RED" "$C_RESET"
}

# ─── Main ────────────────────────────────────────────────────────────────
main() {
    info "Locating a supported Python interpreter..."
    if ! PYTHON_CMD="$(find_python)"; then
        print_python_install_hints
        exit 1
    fi

    # Resolve to absolute path so venv creation records the real binary
    local resolved
    resolved="$("$PYTHON_CMD" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PYTHON_CMD")"
    ok "Using Python: $resolved"

    # Delegate everything else to install.py
    exec "$resolved" "$INSTALL_PY" "$@"
}

main "$@"
