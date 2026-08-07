"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║                    KNOWLEDGE RAG — INSTALLER v3.0                 ║
║      Cross-platform, multi-LLM-client installer (Python core)     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

Installs knowledge-rag (from PyPI or local source) and registers it
as an MCP server in every supported LLM tool detected on the machine:
Claude Code, Claude Desktop, Cursor, Windsurf, VS Code, Cline,
Gemini CLI, Zed.

Runs on Linux, macOS, and Windows from a single codebase. Called by
install.sh and install.ps1 wrappers after Python is available.

Author:  Ailton Rocha (Lyon.)
Version: 3.0.0
Date:    2026-07-02
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Any, Callable

# Force UTF-8 on stdout/stderr so the box-drawing banner survives on Windows
# terminals defaulting to cp1252. Safe no-op on POSIX and on already-UTF-8 shells.
for _stream in (sys.stdout, sys.stderr):
    reconf = getattr(_stream, "reconfigure", None)
    if callable(reconf):
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ============================================================================
# CONFIGURATION
# ============================================================================

SERVER_NAME = "knowledge-rag"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
REQUIREMENTS_FILE = "requirements.txt"
PYPI_PACKAGE = "knowledge-rag"
SUPPORTED_PY = {"3.11", "3.12"}
DEFAULT_INSTALL_DIRNAME = "knowledge-rag"
BACKUP_SUFFIX = ".knowledge-rag.bak"

# ============================================================================
# TERMINAL COLORS (auto-disable when not a TTY or NO_COLOR is set)
# ============================================================================


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if platform.system() == "Windows":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def info(msg: str) -> None:
    print(f"{_c('36', '[*]')} {msg}")


def ok(msg: str) -> None:
    print(f"{_c('32', '[+]')} {msg}")


def warn(msg: str) -> None:
    print(f"{_c('33', '[!]')} {msg}")


def err(msg: str) -> None:
    print(f"{_c('31', '[-]')} {msg}", file=sys.stderr)


def step(title: str) -> None:
    print(f"\n{_c('33', '=== ' + title + ' ===')}")


def skip(msg: str) -> None:
    print(f"{_c('90', '[~]')} {msg}")


# ============================================================================
# BANNER
# ============================================================================

BANNER = r"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   ██╗  ██╗███╗   ██╗ ██████╗ ██╗    ██╗██╗     ███████╗██████╗    ║
    ║   ██║ ██╔╝████╗  ██║██╔═══██╗██║    ██║██║     ██╔════╝██╔══██╗   ║
    ║   █████╔╝ ██╔██╗ ██║██║   ██║██║ █╗ ██║██║     █████╗  ██║  ██║   ║
    ║   ██╔═██╗ ██║╚██╗██║██║   ██║██║███╗██║██║     ██╔══╝  ██║  ██║   ║
    ║   ██║  ██╗██║ ╚████║╚██████╔╝╚███╔███╔╝███████╗███████╗██████╔╝   ║
    ║   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚══╝╚══╝ ╚══════╝╚══════╝╚═════╝    ║
    ║                                                                   ║
    ║              RAG SYSTEM INSTALLER v3.0 — Multi-Client             ║
    ║        Claude · Cursor · Windsurf · VS Code · Gemini · Zed        ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
"""


def print_banner() -> None:
    print(_c("36", BANNER))


# ============================================================================
# PLATFORM HELPERS
# ============================================================================

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def home() -> Path:
    return Path.home()


def appdata() -> Path:
    """Windows-only: %APPDATA%. Empty on other platforms."""
    return Path(os.environ.get("APPDATA", ""))


def vscode_user_dir() -> Path | None:
    """Returns the VS Code user data directory or None if unknown."""
    if IS_WINDOWS:
        base = appdata()
        return base / "Code" / "User" if base else None
    if IS_MACOS:
        return home() / "Library" / "Application Support" / "Code" / "User"
    if IS_LINUX:
        return home() / ".config" / "Code" / "User"
    return None


def claude_desktop_config() -> Path | None:
    if IS_WINDOWS:
        base = appdata()
        return base / "Claude" / "claude_desktop_config.json" if base else None
    if IS_MACOS:
        return home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if IS_LINUX:
        return home() / ".config" / "Claude" / "claude_desktop_config.json"
    return None


def zed_settings() -> Path | None:
    if IS_WINDOWS:
        base = appdata()
        return base / "Zed" / "settings.json" if base else None
    return home() / ".config" / "zed" / "settings.json"


# ============================================================================
# SERVER SPEC BUILDERS (per-client, since schemas differ slightly)
# ============================================================================


def _spec_mcp_servers(install_path: Path, venv_python: Path) -> dict[str, Any]:
    """Standard `mcpServers` schema — Claude Code/Desktop, Cursor, Windsurf, Cline, Gemini."""
    return {
        "type": "stdio",
        "command": str(venv_python),
        "args": ["-m", "mcp_server.server"],
        "cwd": str(install_path),
        "env": {},
    }


def _spec_vscode_servers(install_path: Path, venv_python: Path) -> dict[str, Any]:
    """VS Code uses `servers` (not `mcpServers`) but the entry shape is compatible."""
    return {
        "type": "stdio",
        "command": str(venv_python),
        "args": ["-m", "mcp_server.server"],
        "cwd": str(install_path),
    }


def _spec_zed_context_servers(install_path: Path, venv_python: Path) -> dict[str, Any]:
    """Zed uses `context_servers` with a `source: custom` marker."""
    return {
        "source": "custom",
        "command": str(venv_python),
        "args": ["-m", "mcp_server.server"],
        "env": {},
    }


# ============================================================================
# CLIENT REGISTRY
# ============================================================================


class Client:
    def __init__(
        self,
        key: str,
        display: str,
        path_fn: Callable[[], Path | None],
        json_key: str,
        spec_fn: Callable[[Path, Path], dict[str, Any]],
        detect_hint: str = "",
    ) -> None:
        self.key = key
        self.display = display
        self.path_fn = path_fn
        self.json_key = json_key
        self.spec_fn = spec_fn
        self.detect_hint = detect_hint

    def config_path(self) -> Path | None:
        return self.path_fn()


CLIENTS: list[Client] = [
    Client(
        "claude-code",
        "Claude Code",
        lambda: home() / ".claude.json",
        "mcpServers",
        _spec_mcp_servers,
        "CLI: 'claude'",
    ),
    Client(
        "claude-desktop",
        "Claude Desktop",
        claude_desktop_config,
        "mcpServers",
        _spec_mcp_servers,
        "Desktop app",
    ),
    Client(
        "cursor",
        "Cursor",
        lambda: home() / ".cursor" / "mcp.json",
        "mcpServers",
        _spec_mcp_servers,
        "cursor.com",
    ),
    Client(
        "windsurf",
        "Windsurf (Cascade)",
        lambda: home() / ".codeium" / "windsurf" / "mcp_config.json",
        "mcpServers",
        _spec_mcp_servers,
        "codeium.com/windsurf",
    ),
    Client(
        "vscode",
        "VS Code (Copilot Chat MCP)",
        lambda: (vscode_user_dir() / "mcp.json") if vscode_user_dir() else None,
        "servers",
        _spec_vscode_servers,
        "code.visualstudio.com",
    ),
    Client(
        "cline",
        "Cline (VS Code extension)",
        lambda: (
            (vscode_user_dir() / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json")
            if vscode_user_dir()
            else None
        ),
        "mcpServers",
        _spec_mcp_servers,
        "cline.bot",
    ),
    Client(
        "gemini",
        "Gemini CLI",
        lambda: home() / ".gemini" / "settings.json",
        "mcpServers",
        _spec_mcp_servers,
        "geminicli.com",
    ),
    Client(
        "zed",
        "Zed",
        zed_settings,
        "context_servers",
        _spec_zed_context_servers,
        "zed.dev",
    ),
]


# ============================================================================
# JSON MERGE ENGINE — idempotent, preserves siblings, backs up before writing
# ============================================================================


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        warn(f"{path} contains invalid JSON ({e}) — leaving it untouched")
        return None


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON without BOM, atomically (write to tmp then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        tmp_name = f.name
    os.replace(tmp_name, path)


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    shutil.copy2(path, backup)
    return backup


def register_client(
    client: Client,
    install_path: Path,
    venv_python: Path,
    dry_run: bool,
) -> tuple[bool, str]:
    """
    Idempotently upserts the knowledge-rag entry into a client's config file.
    Returns (changed, message).
    """
    config = client.config_path()
    if config is None:
        return False, f"platform unsupported for {client.display}"

    existing = _read_json(config) or {}
    parent = existing.setdefault(client.json_key, {})
    new_spec = client.spec_fn(install_path, venv_python)

    current_spec = parent.get(SERVER_NAME)
    if current_spec == new_spec:
        return False, f"already registered (no change) — {config}"

    if dry_run:
        return True, f"WOULD write to {config}"

    _backup(config)
    parent[SERVER_NAME] = new_spec
    _write_json_atomic(config, existing)
    return True, f"registered → {config}"


# ============================================================================
# PYTHON DETECTION
# ============================================================================


def python_version_short(exe: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            [str(exe), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def find_python() -> Path:
    step("PYTHON DETECTION")

    candidates: list[str] = []

    # Explicit versioned binaries first
    for v in ("3.12", "3.11"):
        candidates.append(f"python{v}")
        if IS_LINUX:
            candidates += [f"/usr/bin/python{v}", f"/usr/local/bin/python{v}"]
        if IS_MACOS:
            candidates += [
                f"/opt/homebrew/bin/python{v}",
                f"/usr/local/opt/python@{v}/bin/python{v}",
                f"/usr/local/bin/python{v}",
            ]
        if IS_WINDOWS:
            local = os.environ.get("LOCALAPPDATA", "")
            major_minor_flat = v.replace(".", "")
            candidates += [
                f"{local}\\Programs\\Python\\Python{major_minor_flat}\\python.exe",
                f"C:\\Program Files\\Python{major_minor_flat}\\python.exe",
                f"C:\\Python{major_minor_flat}\\python.exe",
            ]

    # Windows py launcher
    if IS_WINDOWS:
        candidates += ["py -3.12", "py -3.11"]

    # Generic fallbacks (checked last)
    candidates += ["python3", "python"]

    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)

        # Handle "py -3.12" split form
        parts = cand.split()
        exe = shutil.which(parts[0]) if not os.path.isabs(parts[0]) else parts[0]
        if not exe or not Path(exe).exists():
            continue
        cmd = [exe] + parts[1:]

        try:
            out = subprocess.check_output(
                cmd + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            ver = out.decode().strip()
        except Exception:
            continue

        if ver in SUPPORTED_PY:
            resolved = (
                subprocess.check_output(
                    cmd + ["-c", "import sys; print(sys.executable)"],
                    stderr=subprocess.STDOUT,
                    timeout=10,
                )
                .decode()
                .strip()
            )
            ok(f"Python {ver} found: {resolved}")
            return Path(resolved)

    err(f"No supported Python found. Need one of: {sorted(SUPPORTED_PY)}")
    _print_python_install_hints()
    sys.exit(1)


def _print_python_install_hints() -> None:
    print("\nInstall a supported Python:")
    if IS_LINUX:
        if shutil.which("apt"):
            print("  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv python3.12-dev")
        if shutil.which("dnf"):
            print("  Fedora/RHEL:   sudo dnf install python3.12 python3.12-devel")
        if shutil.which("pacman"):
            print("  Arch:          sudo pacman -S python")
        if shutil.which("apk"):
            print("  Alpine:        sudo apk add python3 py3-pip")
    elif IS_MACOS:
        print("  macOS:         brew install python@3.12")
    elif IS_WINDOWS:
        print("  Windows:       winget install Python.Python.3.12")
        print("                 (or download from python.org)")
    print("  Any platform:  pyenv install 3.12 && pyenv global 3.12")
    print(_c("31", "\nNOTE: Python 3.13+ is NOT supported (onnxruntime incompatibility)"))


# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

DIRS = [
    "mcp_server",
    "documents",
    "documents/security",
    "documents/logscale",
    "documents/development",
    "documents/general",
    "documents/aar",
    "data",
    "data/chroma_db",
    ".claude",
]


def setup_project_structure(install_path: Path) -> None:
    step("PROJECT STRUCTURE")
    install_path.mkdir(parents=True, exist_ok=True)
    ok(f"Directory ready: {install_path}")
    for d in DIRS:
        (install_path / d).mkdir(parents=True, exist_ok=True)
    ok("Subdirectories created")


# ============================================================================
# VIRTUAL ENVIRONMENT
# ============================================================================


def venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if IS_WINDOWS else "bin") / ("python.exe" if IS_WINDOWS else "python")


def setup_venv(
    install_path: Path,
    python: Path,
    force: bool,
    from_source: bool,
    pypi_version: str | None,
) -> Path:
    step("VIRTUAL ENVIRONMENT")

    venv_dir = install_path / "venv"
    venv_py = venv_python_path(venv_dir)

    if force and venv_dir.exists():
        warn(f"Removing existing venv (--force): {venv_dir}")
        shutil.rmtree(venv_dir)

    if not venv_py.exists():
        info(f"Creating virtual environment with {python} ...")
        venv.EnvBuilder(with_pip=True, upgrade_deps=False, clear=False).create(venv_dir)
        ok("Virtual environment created")
    else:
        ok("Virtual environment exists")

    info("Upgrading pip ...")
    subprocess.check_call([str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    if from_source:
        req_here = Path.cwd() / REQUIREMENTS_FILE
        req_installed = install_path / REQUIREMENTS_FILE
        req = req_installed if req_installed.exists() else req_here
        if not req.exists():
            err(f"{REQUIREMENTS_FILE} not found in {install_path} nor {Path.cwd()}")
            err("Clone the repo first, or drop --from-source to install from PyPI")
            sys.exit(1)
        info(f"Installing from {req} ...")
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "-r", str(req), "--quiet"])
    else:
        target = PYPI_PACKAGE if not pypi_version else f"{PYPI_PACKAGE}=={pypi_version}"
        info(f"Installing {target} from PyPI ...")
        subprocess.check_call([str(venv_py), "-m", "pip", "install", target, "--quiet"])

    ok("Dependencies installed")
    return venv_py


# ============================================================================
# INIT + EMBEDDING MODEL PRE-DOWNLOAD
# ============================================================================


def run_init(install_path: Path, venv_py: Path) -> None:
    step("PROJECT INITIALIZATION")
    info("Exporting config template and presets ...")
    try:
        subprocess.check_call(
            [str(venv_py), "-m", "mcp_server.server", "init"],
            cwd=str(install_path),
        )
        ok("Config template + presets exported")
    except subprocess.CalledProcessError as e:
        warn(f"init returned non-zero exit ({e.returncode}) — continuing")


def preload_embedding_model(venv_py: Path) -> None:
    step("EMBEDDING MODEL (FastEmbed)")
    info(f"Pre-downloading {EMBEDDING_MODEL} (~130 MB, cached in ~/.cache/fastembed/) ...")
    code = f"from fastembed import TextEmbedding; TextEmbedding('{EMBEDDING_MODEL}')"
    try:
        subprocess.check_call([str(venv_py), "-c", code])
        ok(f"Model '{EMBEDDING_MODEL}' cached")
    except subprocess.CalledProcessError:
        warn("Pre-download failed — server will retry on first start")


# ============================================================================
# CLIENT REGISTRATION ORCHESTRATOR
# ============================================================================


def register_all_clients(
    install_path: Path,
    venv_py: Path,
    only: set[str] | None,
    exclude: set[str],
    dry_run: bool,
) -> None:
    step("MCP CLIENT REGISTRATION")

    changed_any = False
    for client in CLIENTS:
        if only and client.key not in only:
            skip(f"{client.display} (not in --for)")
            continue
        if client.key in exclude:
            skip(f"{client.display} (excluded)")
            continue

        config_path = client.config_path()
        if config_path is None:
            skip(f"{client.display} — no known path on this OS")
            continue

        # A client "counts as installed" if its config file OR parent dir exists,
        # OR if --for named it explicitly (user opt-in overrides detection).
        detected = config_path.exists() or config_path.parent.exists() or (only is not None and client.key in only)
        if not detected:
            skip(f"{client.display} — not detected ({client.detect_hint})")
            continue

        try:
            changed, msg = register_client(client, install_path, venv_py, dry_run)
        except Exception as e:
            err(f"{client.display}: {e}")
            continue

        if changed:
            ok(f"{client.display}: {msg}")
            changed_any = True
        else:
            skip(f"{client.display}: {msg}")

    if not changed_any:
        warn("No clients were updated. Use --for <name> to force a specific client.")


# ============================================================================
# SUMMARY
# ============================================================================


def show_summary(install_path: Path, venv_py: Path, py_version: str, dry_run: bool) -> None:
    print()
    print(_c("32", "    ╔═══════════════════════════════════════════════════════════════════╗"))
    print(
        _c(
            "32",
            "    ║                    INSTALLATION COMPLETE!                         ║"
            if not dry_run
            else "    ║                    DRY-RUN COMPLETE                               ║",
        )
    )
    print(_c("32", "    ╚═══════════════════════════════════════════════════════════════════╝"))
    print()
    print(f"    {_c('1', 'Install path:')}   {install_path}")
    print(f"    {_c('1', 'Python (venv):')} {venv_py}  ({py_version})")
    print(f"    {_c('1', 'Embedding:')}     {EMBEDDING_MODEL} (FastEmbed in-process)")
    print()
    print(_c("36", "    NEXT STEPS"))
    print("    " + "─" * 60)
    print(f"    1. Drop documents into {install_path / 'documents'}/")
    print("       (security / logscale / development / aar / general)")
    print("    2. Restart the LLM client(s) — they'll auto-load knowledge-rag")
    print("    3. 13 MCP tools become available on connect")
    print()


# ============================================================================
# ARG PARSING
# ============================================================================


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="install.py",
        description="Cross-platform installer for knowledge-rag (multi-LLM-client).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported clients: " + ", ".join(c.key for c in CLIENTS) + "\n"
            "By default the installer detects which clients are present and registers\n"
            "knowledge-rag in each. Use --for/--exclude to narrow the target set."
        ),
    )
    p.add_argument(
        "--install-path",
        type=Path,
        default=None,
        help=f"Where to install (default: ~/{DEFAULT_INSTALL_DIRNAME})",
    )
    p.add_argument(
        "--from-source",
        action="store_true",
        help="Install from local requirements.txt instead of PyPI",
    )
    p.add_argument(
        "--pypi-version",
        default=None,
        help="Pin PyPI version (e.g. 4.3.1). Ignored with --from-source.",
    )
    p.add_argument("--force", action="store_true", help="Recreate the venv from scratch")
    p.add_argument(
        "--for",
        dest="for_clients",
        default="",
        help="Comma-separated allowlist of clients to configure (default: auto-detect all)",
    )
    p.add_argument(
        "--exclude",
        dest="exclude_clients",
        default="",
        help="Comma-separated blocklist of clients to skip",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    p.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip the embedding model pre-download",
    )
    p.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip 'mcp_server.server init' (config template + presets)",
    )
    p.add_argument(
        "--list-clients",
        action="store_true",
        help="Print the supported clients + their config paths and exit",
    )
    return p.parse_args(argv)


def print_client_registry() -> None:
    print_banner()
    print("Supported clients:\n")
    for c in CLIENTS:
        path = c.config_path()
        print(f"  {_c('1', c.key):<24} {c.display}")
        print(f"    path: {path if path else '(not available on this OS)'}")
        print(f"    key:  {c.json_key}")
        print()


# ============================================================================
# MAIN
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.list_clients:
        print_client_registry()
        return 0

    print_banner()

    install_path: Path = args.install_path or (home() / DEFAULT_INSTALL_DIRNAME)
    install_path = install_path.expanduser().resolve()

    # If --from-source and cwd looks like the repo, prefer it (unless user overrode)
    if args.from_source and args.install_path is None:
        cwd = Path.cwd()
        if (cwd / "mcp_server" / "server.py").exists() and (cwd / REQUIREMENTS_FILE).exists():
            install_path = cwd
            info(f"Detected source checkout — using current directory: {install_path}")

    info(f"Install path: {install_path}")
    info(f"Platform:     {platform.system()} ({platform.machine()})")
    info(f"Mode:         {'from-source' if args.from_source else 'PyPI'}")
    if args.dry_run:
        warn("Dry-run mode — no files will be written")

    only_raw = {s.strip() for s in args.for_clients.split(",") if s.strip()}
    exclude = {s.strip() for s in args.exclude_clients.split(",") if s.strip()}

    known_keys = {c.key for c in CLIENTS}
    for bad in (only_raw | exclude) - known_keys:
        err(f"Unknown client: {bad!r} (see --list-clients)")
        return 2
    only = only_raw or None

    # 1. Python
    python = find_python()
    py_version = python_version_short(python) or "?"

    # 2. Project layout
    setup_project_structure(install_path)

    # 3. Venv + install
    if args.dry_run:
        skip("Skipping venv creation in dry-run mode")
        venv_py = venv_python_path(install_path / "venv")
    else:
        venv_py = setup_venv(
            install_path,
            python,
            force=args.force,
            from_source=args.from_source,
            pypi_version=args.pypi_version,
        )

    # 4. Init
    if not args.skip_init and not args.dry_run:
        run_init(install_path, venv_py)

    # 5. Embedding model
    if not args.skip_model and not args.dry_run:
        preload_embedding_model(venv_py)

    # 6. Client registration
    register_all_clients(install_path, venv_py, only=only, exclude=exclude, dry_run=args.dry_run)

    # 7. Summary
    show_summary(install_path, venv_py, py_version, dry_run=args.dry_run)
    ok("Done." if not args.dry_run else "Dry-run finished — re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        err("Interrupted by user")
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        err(f"Subprocess failed with exit {e.returncode}: {' '.join(map(str, e.cmd))}")
        sys.exit(e.returncode or 1)
