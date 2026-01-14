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
    .\install.ps1 -SkipPython        # Skip Python installation
    .\install.ps1 -SkipOllama        # Skip Ollama installation
    .\install.ps1 -DocsPath "C:\Docs" # Custom documents path
#>

[CmdletBinding()]
param(
    [switch]$SkipPython,
    [switch]$SkipOllama,
    [switch]$SkipIndex,
    [string]$InstallPath = "$env:USERPROFILE\OneDrive\Documentos\AI-Agents\knowledge-rag",
    [string]$DocsPath = "",
    [switch]$Force
)

# ============================================================================
# CONFIGURATION
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$CONFIG = @{
    PythonVersion = "3.12"
    PythonMinVersion = "3.11"
    PythonInstallerUrl = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
    OllamaInstallerUrl = "https://ollama.com/download/OllamaSetup.exe"
    EmbeddingModel = "nomic-embed-text"
    RequiredPackages = @("chromadb", "pymupdf", "ollama", "mcp")
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

function Get-PythonPath {
    # Check common Python installation paths
    $paths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )

    foreach ($path in $paths) {
        if (Test-Path $path) {
            return $path
        }
    }

    # Try to find via py launcher
    try {
        $pyPath = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($pyPath -and (Test-Path $pyPath)) {
            return $pyPath
        }

        $pyPath = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($pyPath -and (Test-Path $pyPath)) {
            return $pyPath
        }
    } catch {}

    # Try PATH
    try {
        $pyPath = (Get-Command python -ErrorAction SilentlyContinue).Source
        if ($pyPath) {
            $version = & $pyPath --version 2>&1
            if ($version -match "3\.(11|12)") {
                return $pyPath
            }
        }
    } catch {}

    return $null
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

function Install-Python {
    Write-Host "`n=== PYTHON INSTALLATION ===" -ForegroundColor Yellow

    $pythonPath = Get-PythonPath

    if ($pythonPath) {
        $version = & $pythonPath --version 2>&1
        Write-Step "Python found: $version at $pythonPath" "OK"
        return $pythonPath
    }

    Write-Step "Python 3.11/3.12 not found. Installing..." "WARN"

    $installerPath = "$env:TEMP\python-installer.exe"

    Write-Step "Downloading Python installer..." "INFO"
    Invoke-WebRequest -Uri $CONFIG.PythonInstallerUrl -OutFile $installerPath

    Write-Step "Running Python installer (this may take a few minutes)..." "INFO"
    $installArgs = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait

    Remove-Item $installerPath -Force -ErrorAction SilentlyContinue

    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    $pythonPath = Get-PythonPath
    if ($pythonPath) {
        Write-Step "Python installed successfully!" "OK"
        return $pythonPath
    } else {
        throw "Python installation failed. Please install Python 3.11 or 3.12 manually."
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
    $dirs = @(
        "mcp_server",
        "documents",
        "documents\security",
        "documents\logscale",
        "documents\development",
        "documents\general",
        "chroma_db",
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

function Setup-VirtualEnvironment {
    param([string]$PythonPath)

    Write-Host "`n=== VIRTUAL ENVIRONMENT ===" -ForegroundColor Yellow

    $venvPath = Join-Path $InstallPath "venv"
    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    $venvPip = Join-Path $venvPath "Scripts\pip.exe"

    # Create venv if needed
    if (-not (Test-Path $venvPython) -or $Force) {
        Write-Step "Creating virtual environment..." "INFO"
        & $PythonPath -m venv $venvPath
        Write-Step "Virtual environment created" "OK"
    } else {
        Write-Step "Virtual environment exists" "OK"
    }

    # Upgrade pip
    Write-Step "Upgrading pip..." "INFO"
    & $venvPython -m pip install --upgrade pip --quiet

    # Install packages
    Write-Step "Installing dependencies..." "INFO"
    foreach ($package in $CONFIG.RequiredPackages) {
        Write-Step "  Installing $package..." "INFO"
        & $venvPip install $package --quiet
    }

    Write-Step "All dependencies installed!" "OK"

    return $venvPython
}

function Create-SourceFiles {
    Write-Host "`n=== SOURCE FILES ===" -ForegroundColor Yellow

    # __init__.py
    $initPath = Join-Path $InstallPath "mcp_server\__init__.py"
    if (-not (Test-Path $initPath)) {
        '"""Knowledge RAG MCP Server Package"""' | Out-File -FilePath $initPath -Encoding utf8
    }

    # config.py
    $configContent = @'
"""Configuration for Knowledge RAG System"""

from pathlib import Path
from typing import Dict, List

class Config:
    """Configuration settings"""

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "chroma_db"
    docs_dir: Path = base_dir / "documents"
    chroma_dir: Path = data_dir

    # ChromaDB
    collection_name: str = "knowledge_base"

    # Ollama
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Search
    default_results: int = 5
    max_results: int = 20

    # Supported formats
    supported_formats: List[str] = [".md", ".txt", ".pdf", ".py", ".json"]

    # Keyword routing
    keyword_routes: Dict[str, List[str]] = {
        "security": ["exploit", "vulnerability", "pentest", "attack", "malware",
                     "reverse", "shellcode", "buffer overflow", "injection", "xss",
                     "sqli", "rce", "privilege escalation", "lateral movement"],
        "ctf": ["ctf", "flag", "challenge", "hackthebox", "htb", "tryhackme",
                "picoctf", "pwn", "crypto", "forensics", "stego"],
        "logscale": ["logscale", "humio", "cql", "lql", "crowdstrike", "falcon",
                     "formatTime", "groupBy", "regex", "case{}", "detection"],
        "development": ["python", "javascript", "typescript", "react", "api",
                        "database", "docker", "kubernetes", "ci/cd", "git"]
    }

config = Config()
'@

    $configPath = Join-Path $InstallPath "mcp_server\config.py"
    $configContent | Out-File -FilePath $configPath -Encoding utf8
    Write-Step "Created: config.py" "OK"

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
    param([string]$VenvPython)

    Write-Host "`n=== MCP CONFIGURATION ===" -ForegroundColor Yellow

    $mcpConfig = @{
        mcpServers = @{
            "knowledge-rag" = @{
                type = "stdio"
                command = $VenvPython.Replace("\", "\\")
                args = @("-m", "mcp_server.server")
                cwd = $InstallPath.Replace("\", "\\")
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
    param([string]$VenvPython)

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
import sys
sys.path.insert(0, r'$InstallPath')
from mcp_server.server import KnowledgeOrchestrator

orch = KnowledgeOrchestrator()
result = orch.index_all()
print(f"Indexed: {result['indexed']} files, {result['chunks_added']} chunks")
print(f"Categories: {result['categories']}")
"@

    try {
        $result = & $VenvPython -c $indexScript 2>&1
        Write-Host $result
        Write-Step "Indexing complete!" "OK"
    } catch {
        Write-Step "Indexing failed: $_" "ERROR"
    }
}

function Show-Summary {
    param([string]$VenvPython)

    $summary = @"

    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    INSTALLATION COMPLETE!                         ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Installation Path: $InstallPath
    Python:            $VenvPython
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
    if (-not $SkipPython -or -not $SkipOllama) {
        if (-not (Test-Administrator)) {
            Write-Step "Some installations may require administrator privileges" "WARN"
        }
    }

    # Step 1: Python
    if ($SkipPython) {
        Write-Step "Skipping Python installation" "SKIP"
        $pythonPath = Get-PythonPath
        if (-not $pythonPath) {
            throw "Python 3.11/3.12 not found. Run without -SkipPython"
        }
    } else {
        $pythonPath = Install-Python
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

    # Step 4: Virtual environment
    $venvPython = Setup-VirtualEnvironment -PythonPath $pythonPath

    # Step 5: Source files
    Create-SourceFiles

    # Step 6: MCP configuration
    Setup-MCPConfiguration -VenvPython $venvPython

    # Step 7: Initial indexing
    if (-not $SkipIndex) {
        Run-InitialIndex -VenvPython $venvPython
    } else {
        Write-Step "Skipping initial indexing" "SKIP"
    }

    # Summary
    Show-Summary -VenvPython $venvPython

    Write-Host "Installation completed successfully!" -ForegroundColor Green

} catch {
    Write-Host "`n[!] Installation failed: $_" -ForegroundColor Red
    Write-Host "Please check the error and try again." -ForegroundColor Yellow
    exit 1
}
