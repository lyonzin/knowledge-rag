<#
.SYNOPSIS
    knowledge-rag skills installer for Windows PowerShell 5.1 and PowerShell 7.

.DESCRIPTION
    A thin wrapper around `knowledge-rag skills`, which detects your AI coding
    assistants and asks before installing into each one's skills folder. With
    no subcommand it runs `install`. Every option goes to `knowledge-rag skills`;
    run `knowledge-rag skills install --help` for the list. The exit code is the
    one of `knowledge-rag skills`, 127 when knowledge-rag is not installed, and
    1 when it is too old to have the skills installer.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File skills\install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File skills\install.ps1 status

.EXAMPLE
    irm https://raw.githubusercontent.com/lyonzin/knowledge-rag/master/skills/install.ps1 | iex
#>

# No param() block: advanced parameter binding would reject the double-dash
# options before they reach knowledge-rag.
$forwarded = @($args)
if ($forwarded.Count -eq 0 -or $forwarded[0] -notin @("install", "uninstall", "status", "-h", "--help")) {
    $forwarded = @("install") + $forwarded
}

function Find-Application([string]$Name) {
    Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
}

function Write-Help([string[]]$Lines) {
    foreach ($line in $Lines) { [Console]::Error.WriteLine($line) }
}

# knowledge-rag-skills runs `knowledge-rag skills`. Releases up to 5.0.0 lack it, and their
# `knowledge-rag` would ignore `skills` and start the MCP server, so never call that one.
$installer = Find-Application "knowledge-rag-skills"
if ($installer) {
    & $installer.Source @forwarded
    $code = $LASTEXITCODE
} elseif (Find-Application "knowledge-rag") {
    Write-Help @(
        "This knowledge-rag is too old for the skills installer. Upgrade it, then run this again:",
        "  pip install -U knowledge-rag    (or: pipx upgrade knowledge-rag, uv tool upgrade knowledge-rag)"
    )
    $code = 1
} else {
    Write-Help @(
        "knowledge-rag is not installed, or not on PATH. Install it, then run this again:",
        "  pip install knowledge-rag    (or: pipx install knowledge-rag, uv tool install knowledge-rag)",
        "Or run the same installer through npm, which needs Node.js and Python 3.11 or later:",
        "  npx -y knowledge-rag skills install",
        "Installed it in a virtual environment? Activate that environment first."
    )
    $code = 127
}

# Under `irm ... | iex` this runs in the caller's session, which exit would close.
if ($PSCommandPath) { exit $code }
