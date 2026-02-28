<#
.SYNOPSIS
    Knowledge RAG System - Automated Installation Script

.DESCRIPTION
    Installs and configures a local RAG (Retrieval-Augmented Generation) system
    with ChromaDB, Ollama embeddings, and MCP integration for Claude Code.

.AUTHOR
    Ailton Rocha (Lyon) - AI Operator

.VERSION
    1.0.0

.REQUIREMENTS
    - Windows 10/11
    - Internet connection
    - Administrator privileges (for some installations)

.USAGE
    .\install.ps1                    # Full installation
    .\install.ps1 -SkipUv            # Skip uv installation
    .\install.ps1 -SkipOllama        # Skip Ollama installation
    .\install.ps1 -DocsPath "C:\Docs" # Custom documents path
#>

[CmdletBinding()]
param(
    [switch]$SkipUv,
    [switch]$SkipOllama,
    [switch]$SkipIndex,
    [string]$InstallPath = $PSScriptRoot,  # Uses directory where script is located
    [string]$DocsPath = "",
    [switch]$Force
)

# ============================================================================
# CONFIGURATION
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$CONFIG = @{
    UvInstallerUrl = "https://astral.sh/uv/install.ps1"
    OllamaInstallerUrl = "https://ollama.com/download/OllamaSetup.exe"
    EmbeddingModel = "nomic-embed-text"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Write-Banner {
    $banner = @"

    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   ██╗  ██╗███╗   ██╗ ██████╗ ██╗    ██╗██╗     ███████╗██████╗   ║
    ║   ██║ ██╔╝████╗  ██║██╔═══██╗██║    ██║██║     ██╔════╝██╔══██╗  ║
    ║   █████╔╝ ██╔██╗ ██║██║   ██║██║ █╗ ██║██║     █████╗  ██║  ██║  ║
    ║   ██╔═██╗ ██║╚██╗██║██║   ██║██║███╗██║██║     ██╔══╝  ██║  ██║  ║
    ║   ██║  ██╗██║ ╚████║╚██████╔╝╚███╔███╔╝███████╗███████╗██████╔╝  ║
    ║   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚══╝╚══╝ ╚══════╝╚══════╝╚═════╝   ║
    ║                                                                   ║
    ║                    RAG SYSTEM INSTALLER v1.0                      ║
    ║              Local Semantic Search for Claude Code                ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝

"@
    Write-Host $banner -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message, [string]$Status = "INFO")

    $colors = @{
        "INFO" = "Cyan"
        "OK" = "Green"
        "WARN" = "Yellow"
        "ERROR" = "Red"
        "SKIP" = "DarkGray"
    }

    $symbols = @{
        "INFO" = "[*]"
        "OK" = "[+]"
        "WARN" = "[!]"
        "ERROR" = "[-]"
        "SKIP" = "[~]"
    }

    Write-Host "$($symbols[$Status]) " -ForegroundColor $colors[$Status] -NoNewline
    Write-Host $Message
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-UvPath {
    try {
        return (Get-Command uv -ErrorAction SilentlyContinue).Source
    } catch {
        return $null
    }
}

function Get-OllamaPath {
    $paths = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:PROGRAMFILES\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe"
    )

    foreach ($path in $paths) {
        if (Test-Path $path) {
            return $path
        }
    }

    try {
        return (Get-Command ollama -ErrorAction SilentlyContinue).Source
    } catch {
        return $null
    }
}

function Test-OllamaRunning {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

function Start-OllamaService {
    $ollamaPath = Get-OllamaPath
    if (-not $ollamaPath) {
        return $false
    }

    Write-Step "Starting Ollama service..." "INFO"
    Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden

    # Wait for service to start
    $maxWait = 30
    $waited = 0
    while (-not (Test-OllamaRunning) -and $waited -lt $maxWait) {
        Start-Sleep -Seconds 1
        $waited++
    }

    return (Test-OllamaRunning)
}

# ============================================================================
# INSTALLATION STEPS
# ============================================================================

function Install-Uv {
    Write-Host "`n=== UV INSTALLATION ===" -ForegroundColor Yellow

    $uvPath = Get-UvPath

    if ($uvPath) {
        $version = & $uvPath --version 2>&1
        Write-Step "uv found: $version" "OK"
        return $uvPath
    }

    Write-Step "uv not found. Installing..." "WARN"

    Write-Step "Downloading and running uv installer..." "INFO"
    & ([scriptblock]::Create((Invoke-RestMethod $CONFIG.UvInstallerUrl)))

    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    $uvPath = Get-UvPath
    if ($uvPath) {
        $version = & $uvPath --version 2>&1
        Write-Step "uv installed successfully: $version" "OK"
        return $uvPath
    } else {
        throw "uv installation failed. Please install uv manually: https://docs.astral.sh/uv/"
    }
}

function Install-Ollama {
    Write-Host "`n=== OLLAMA INSTALLATION ===" -ForegroundColor Yellow

    $ollamaPath = Get-OllamaPath

    if ($ollamaPath) {
        Write-Step "Ollama found at $ollamaPath" "OK"
        return $ollamaPath
    }

    Write-Step "Ollama not found. Installing..." "WARN"

    $installerPath = "$env:TEMP\OllamaSetup.exe"

    Write-Step "Downloading Ollama installer..." "INFO"
    Invoke-WebRequest -Uri $CONFIG.OllamaInstallerUrl -OutFile $installerPath

    Write-Step "Running Ollama installer..." "INFO"
    Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait

    Remove-Item $installerPath -Force -ErrorAction SilentlyContinue

    # Wait for installation to complete
    Start-Sleep -Seconds 5

    $ollamaPath = Get-OllamaPath
    if ($ollamaPath) {
        Write-Step "Ollama installed successfully!" "OK"
        return $ollamaPath
    } else {
        throw "Ollama installation failed. Please install Ollama manually from https://ollama.com"
    }
}

function Install-EmbeddingModel {
    param([string]$OllamaPath)

    Write-Host "`n=== EMBEDDING MODEL ===" -ForegroundColor Yellow

    # Ensure Ollama is running
    if (-not (Test-OllamaRunning)) {
        if (-not (Start-OllamaService)) {
            throw "Could not start Ollama service. Please start it manually."
        }
    }

    Write-Step "Ollama service is running" "OK"

    # Check if model exists
    try {
        $models = & $OllamaPath list 2>&1
        if ($models -match $CONFIG.EmbeddingModel) {
            Write-Step "Model '$($CONFIG.EmbeddingModel)' already installed" "OK"
            return
        }
    } catch {}

    Write-Step "Pulling embedding model '$($CONFIG.EmbeddingModel)'..." "INFO"
    Write-Step "This may take a few minutes on first run..." "INFO"

    & $OllamaPath pull $CONFIG.EmbeddingModel

    if ($LASTEXITCODE -eq 0) {
        Write-Step "Embedding model installed successfully!" "OK"
    } else {
        throw "Failed to pull embedding model"
    }
}

function Setup-ProjectStructure {
    Write-Host "`n=== PROJECT STRUCTURE ===" -ForegroundColor Yellow

    # Create main directory
    if (-not (Test-Path $InstallPath)) {
        New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
        Write-Step "Created: $InstallPath" "OK"
    } else {
        Write-Step "Directory exists: $InstallPath" "OK"
    }

    # Create subdirectories
    # Note: documents/ is optional - add your own documents there
    # The config.py expects data/chroma_db/ for vector storage
    $dirs = @(
        "mcp_server",
        "documents",
        "documents\security",
        "documents\logscale",
        "documents\development",
        "documents\general",
        "data",
        "data\chroma_db",
        ".claude"
    )

    foreach ($dir in $dirs) {
        $fullPath = Join-Path $InstallPath $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }

    Write-Step "Directory structure created" "OK"
}

function Install-Dependencies {
    param([string]$UvPath)

    Write-Host "`n=== DEPENDENCIES ===" -ForegroundColor Yellow

    Write-Step "Running uv sync (installs Python & dependencies)..." "INFO"
    Push-Location $InstallPath
    try {
        & $UvPath sync
        if ($LASTEXITCODE -eq 0) {
            Write-Step "All dependencies installed!" "OK"
        } else {
            throw "uv sync failed"
        }
    } finally {
        Pop-Location
    }
}

function Create-SourceFiles {
    Write-Host "`n=== SOURCE FILES ===" -ForegroundColor Yellow

    # __init__.py
    $initPath = Join-Path $InstallPath "mcp_server\__init__.py"
    if (-not (Test-Path $initPath)) {
        '"""Knowledge RAG MCP Server Package"""' | Out-File -FilePath $initPath -Encoding utf8
    }

    # config.py - only create if doesn't exist (preserve git version)
    $configPath = Join-Path $InstallPath "mcp_server\config.py"
    if (Test-Path $configPath) {
        Write-Step "Found: config.py (using existing)" "OK"
    } else {
        Write-Step "config.py not found - please ensure you cloned the repository" "WARN"
    }

    # Check if ingestion.py and server.py exist
    $ingestionPath = Join-Path $InstallPath "mcp_server\ingestion.py"
    $serverPath = Join-Path $InstallPath "mcp_server\server.py"

    if (-not (Test-Path $ingestionPath)) {
        Write-Step "ingestion.py not found - please copy from source" "WARN"
    } else {
        Write-Step "Found: ingestion.py" "OK"
    }

    if (-not (Test-Path $serverPath)) {
        Write-Step "server.py not found - please copy from source" "WARN"
    } else {
        Write-Step "Found: server.py" "OK"
    }
}

function Setup-MCPConfiguration {
    Write-Host "`n=== MCP CONFIGURATION ===" -ForegroundColor Yellow

    # Use cmd /c wrapper to ensure working directory is set correctly
    # (Claude Code may not respect the cwd property)
    $escapedPath = $InstallPath.Replace("\", "\\")

    $mcpConfig = @{
        mcpServers = @{
            "knowledge-rag" = @{
                type = "stdio"
                command = "cmd"
                args = @("/c", "cd /d $escapedPath && uv run -m mcp_server.server")
                env = @{}
            }
        }
    }

    $mcpJson = $mcpConfig | ConvertTo-Json -Depth 10

    # Project-level config
    $projectMcpPath = Join-Path $InstallPath ".claude\mcp.json"
    $mcpJson | Out-File -FilePath $projectMcpPath -Encoding utf8
    Write-Step "Created: .claude\mcp.json (project)" "OK"

    # Global config
    $globalClaudeDir = "$env:USERPROFILE\.claude"
    if (-not (Test-Path $globalClaudeDir)) {
        New-Item -ItemType Directory -Path $globalClaudeDir -Force | Out-Null
    }

    $globalMcpPath = Join-Path $globalClaudeDir "mcp.json"

    # Merge with existing config if present
    if (Test-Path $globalMcpPath) {
        try {
            $existingConfig = Get-Content $globalMcpPath -Raw | ConvertFrom-Json -AsHashtable
            $existingConfig.mcpServers["knowledge-rag"] = $mcpConfig.mcpServers["knowledge-rag"]
            $existingConfig | ConvertTo-Json -Depth 10 | Out-File -FilePath $globalMcpPath -Encoding utf8
            Write-Step "Updated: ~/.claude/mcp.json (global)" "OK"
        } catch {
            $mcpJson | Out-File -FilePath $globalMcpPath -Encoding utf8
            Write-Step "Created: ~/.claude/mcp.json (global)" "OK"
        }
    } else {
        $mcpJson | Out-File -FilePath $globalMcpPath -Encoding utf8
        Write-Step "Created: ~/.claude/mcp.json (global)" "OK"
    }
}

function Run-InitialIndex {
    param([string]$UvPath)

    Write-Host "`n=== INITIAL INDEXING ===" -ForegroundColor Yellow

    # Check if there are documents to index
    $docsPath = Join-Path $InstallPath "documents"
    $docCount = (Get-ChildItem -Path $docsPath -Recurse -File | Where-Object {
        $_.Extension -in @(".md", ".txt", ".pdf", ".py", ".json")
    }).Count

    if ($docCount -eq 0) {
        Write-Step "No documents found to index. Add documents to: $docsPath" "WARN"
        return
    }

    Write-Step "Found $docCount documents to index" "INFO"

    # Ensure Ollama is running
    if (-not (Test-OllamaRunning)) {
        if (-not (Start-OllamaService)) {
            Write-Step "Ollama not running. Start it and run indexing manually." "WARN"
            return
        }
    }

    Write-Step "Running initial indexing..." "INFO"

    $indexScript = @"
from mcp_server.server import KnowledgeOrchestrator

orch = KnowledgeOrchestrator()
result = orch.index_all()
print(f"Indexed: {result['indexed']} files, {result['chunks_added']} chunks")
print(f"Categories: {result['categories']}")
"@

    Push-Location $InstallPath
    try {
        $result = & $UvPath run python -c $indexScript 2>&1
        Write-Host $result
        Write-Step "Indexing complete!" "OK"
    } catch {
        Write-Step "Indexing failed: $_" "ERROR"
    } finally {
        Pop-Location
    }
}

function Show-Summary {
    $summary = @"

    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    INSTALLATION COMPLETE!                         ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Installation Path: $InstallPath
    Package Manager:   uv
    Embedding Model:   $($CONFIG.EmbeddingModel)

    ┌─────────────────────────────────────────────────────────────────┐
    │ NEXT STEPS                                                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │ 1. Add documents to: $InstallPath\documents\
    │    - security\   -> Security/pentest content
    │    - logscale\   -> LogScale/CQL queries
    │    - development\ -> Code/dev documentation
    │    - general\    -> Other documents
    │                                                                  │
    │ 2. Restart Claude Code to load the MCP server                    │
    │                                                                  │
    │ 3. Available MCP Tools:                                          │
    │    - search_knowledge(query, max_results, category)              │
    │    - get_document(filepath)                                      │
    │    - reindex_documents(force)                                    │
    │    - list_categories()                                           │
    │    - list_documents(category)                                    │
    │    - get_index_stats()                                           │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

    IMPORTANT: Ollama must be running before using the RAG system!
    Start Ollama with: ollama serve

"@

    Write-Host $summary -ForegroundColor Green
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

try {
    Write-Banner

    # Check admin for installations
    if (-not $SkipOllama) {
        if (-not (Test-Administrator)) {
            Write-Step "Some installations may require administrator privileges" "WARN"
        }
    }

    # Step 1: uv
    if ($SkipUv) {
        Write-Step "Skipping uv installation" "SKIP"
        $uvPath = Get-UvPath
        if (-not $uvPath) {
            throw "uv not found. Run without -SkipUv"
        }
    } else {
        $uvPath = Install-Uv
    }

    # Step 2: Ollama
    if ($SkipOllama) {
        Write-Step "Skipping Ollama installation" "SKIP"
        $ollamaPath = Get-OllamaPath
    } else {
        $ollamaPath = Install-Ollama
        if ($ollamaPath) {
            Install-EmbeddingModel -OllamaPath $ollamaPath
        }
    }

    # Step 3: Project structure
    Setup-ProjectStructure

    # Step 4: Install dependencies (uv sync handles venv + packages)
    Install-Dependencies -UvPath $uvPath

    # Step 5: Source files
    Create-SourceFiles

    # Step 6: MCP configuration
    Setup-MCPConfiguration

    # Step 7: Initial indexing
    if (-not $SkipIndex) {
        Run-InitialIndex -UvPath $uvPath
    } else {
        Write-Step "Skipping initial indexing" "SKIP"
    }

    # Summary
    Show-Summary

    Write-Host "Installation completed successfully!" -ForegroundColor Green

} catch {
    Write-Host "`n[!] Installation failed: $_" -ForegroundColor Red
    Write-Host "Please check the error and try again." -ForegroundColor Yellow
    exit 1
}
