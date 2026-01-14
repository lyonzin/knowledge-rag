<#
.SYNOPSIS
    Starts Ollama service if not already running

.DESCRIPTION
    Checks if Ollama is running and starts it if needed.
    Also ensures the embedding model is available.
#>

$ErrorActionPreference = "SilentlyContinue"

function Test-OllamaRunning {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

function Get-OllamaPath {
    $paths = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:PROGRAMFILES\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe"
    )
    foreach ($path in $paths) {
        if (Test-Path $path) { return $path }
    }
    return (Get-Command ollama -ErrorAction SilentlyContinue).Source
}

Write-Host "Checking Ollama status..." -ForegroundColor Cyan

if (Test-OllamaRunning) {
    Write-Host "[OK] Ollama is already running" -ForegroundColor Green
} else {
    $ollamaPath = Get-OllamaPath
    if (-not $ollamaPath) {
        Write-Host "[ERROR] Ollama not found. Please install from https://ollama.com" -ForegroundColor Red
        exit 1
    }

    Write-Host "[*] Starting Ollama service..." -ForegroundColor Yellow
    Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden

    $waited = 0
    while (-not (Test-OllamaRunning) -and $waited -lt 30) {
        Start-Sleep -Seconds 1
        $waited++
        Write-Host "." -NoNewline
    }
    Write-Host ""

    if (Test-OllamaRunning) {
        Write-Host "[OK] Ollama started successfully" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to start Ollama" -ForegroundColor Red
        exit 1
    }
}

# Check for embedding model
$ollamaPath = Get-OllamaPath
$models = & $ollamaPath list 2>&1
if ($models -match "nomic-embed-text") {
    Write-Host "[OK] Embedding model 'nomic-embed-text' is available" -ForegroundColor Green
} else {
    Write-Host "[*] Pulling embedding model..." -ForegroundColor Yellow
    & $ollamaPath pull nomic-embed-text
}

Write-Host "`nOllama is ready for Knowledge RAG!" -ForegroundColor Green
