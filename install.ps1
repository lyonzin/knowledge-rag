<#
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║             KNOWLEDGE RAG — INSTALLER v3.0 (PowerShell)           ║
║        Cross-platform, multi-LLM-client — Windows wrapper         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

.SYNOPSIS
    Thin PowerShell wrapper. Finds (or installs) a supported Python 3.11/3.12
    and delegates all installation logic to install.py.

.DESCRIPTION
    All actual work (venv, pip install, MCP client detection, JSON merges,
    embedding model pre-download) lives in install.py so Windows, macOS,
    and Linux share one implementation. This wrapper only:

      1. Locates or installs Python 3.11/3.12 (winget preferred, python.org fallback)
      2. Refreshes PATH so the new python is visible in this session
      3. Runs   python install.py  <args>

    See   .\install.ps1 -- --help   for the full flag list (any arg after `--`
    is forwarded to install.py verbatim).

.PARAMETER SkipPython
    Do not attempt to install Python; only look for an existing installation.

.PARAMETER PythonInstaller
    Override the default Python installer URL (advanced).

.PARAMETER Args
    Remaining args forwarded to install.py. Use `--` to separate them clearly.

.EXAMPLE
    .\install.ps1
    Full install (auto-detects LLM clients, registers knowledge-rag in each).

.EXAMPLE
    .\install.ps1 -- --dry-run --for cursor,claude-code
    Dry-run against Cursor + Claude Code only.

.EXAMPLE
    .\install.ps1 -- --from-source --install-path C:\dev\knowledge-rag
    Install from local source into a custom path.

.NOTES
    Autor:   Ailton Rocha (Lyon.)
    Versao:  3.0.0
    Data:    2026-07-02
#>

[CmdletBinding()]
param(
    [switch]$SkipPython,
    [string]$PythonInstaller = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args = @()
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

# ─── Colors ──────────────────────────────────────────────────────────────
function W-Info  { param([string]$m) Write-Host "[*] " -ForegroundColor Cyan -NoNewline;    Write-Host $m }
function W-Ok    { param([string]$m) Write-Host "[+] " -ForegroundColor Green -NoNewline;   Write-Host $m }
function W-Warn  { param([string]$m) Write-Host "[!] " -ForegroundColor Yellow -NoNewline;  Write-Host $m }
function W-Err   { param([string]$m) Write-Host "[-] " -ForegroundColor Red -NoNewline;     Write-Host $m }

$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$installPy  = Join-Path $scriptDir "install.py"

if (-not (Test-Path $installPy)) {
    W-Err "install.py not found next to install.ps1 (looked at $installPy)"
    exit 1
}

# ─── Python detection ────────────────────────────────────────────────────
function Get-ExistingPython {
    # 1) Well-known install paths
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) {
            $ver = (& $p -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
            if ($ver -eq "3.11" -or $ver -eq "3.12") { return (Resolve-Path $p).Path }
        }
    }

    # 2) py launcher
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11")) {
            try {
                $pyExe = (& py "-$v" -c "import sys;print(sys.executable)" 2>$null).Trim()
                if ($pyExe -and (Test-Path $pyExe)) { return $pyExe }
            } catch {}
        }
    }

    # 3) PATH
    foreach ($name in @("python3.12", "python3.11", "python3", "python")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $ver = (& $cmd.Source -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
            if ($ver -eq "3.11" -or $ver -eq "3.12") { return $cmd.Source }
        }
    }
    return $null
}

function Refresh-Path {
    $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Install-Python312 {
    W-Info "Attempting to install Python 3.12 ..."

    # Prefer winget when available (no admin needed for user scope)
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        W-Info "winget detected — installing Python.Python.3.12 (user scope)"
        try {
            & winget install --id Python.Python.3.12 --scope user --silent `
                --accept-package-agreements --accept-source-agreements | Out-Null
            Refresh-Path
            $exe = Get-ExistingPython
            if ($exe) { return $exe }
            W-Warn "winget completed but Python still not detected; falling back to python.org"
        } catch {
            W-Warn "winget install failed: $($_.Exception.Message) — falling back to python.org"
        }
    }

    # Fallback: python.org offline installer
    $tmp = Join-Path $env:TEMP "python-installer-$([guid]::NewGuid().ToString('N')).exe"
    W-Info "Downloading Python installer from $PythonInstaller ..."
    Invoke-WebRequest -Uri $PythonInstaller -OutFile $tmp -UseBasicParsing

    W-Info "Running Python installer (per-user, silent) ..."
    Start-Process -FilePath $tmp -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    Refresh-Path
    $exe = Get-ExistingPython
    if (-not $exe) { throw "Python installation completed but no supported interpreter was found." }
    return $exe
}

# ─── Main ────────────────────────────────────────────────────────────────
try {
    W-Info "Locating a supported Python interpreter (3.11 or 3.12) ..."
    $python = Get-ExistingPython

    if (-not $python) {
        if ($SkipPython) {
            W-Err "No supported Python found and -SkipPython was set."
            W-Info "Install manually: winget install Python.Python.3.12"
            exit 1
        }
        $python = Install-Python312
    }
    W-Ok "Using Python: $python"

    # Forward every remaining flag to install.py verbatim
    W-Info "Delegating to install.py ..."
    & $python $installPy @Args
    $code = $LASTEXITCODE
    if ($code -ne 0) { exit $code }

} catch {
    W-Err "Installation failed: $($_.Exception.Message)"
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    exit 1
}
