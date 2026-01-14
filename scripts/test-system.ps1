<#
.SYNOPSIS
    Tests the Knowledge RAG system

.DESCRIPTION
    Runs a series of tests to verify the system is working correctly.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Write-Host "`n=== KNOWLEDGE RAG SYSTEM TEST ===" -ForegroundColor Cyan

# Test 1: Check Python
Write-Host "`n[1/5] Checking Python..." -ForegroundColor Yellow
if (Test-Path $VenvPython) {
    $version = & $VenvPython --version 2>&1
    Write-Host "  [OK] $version" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Virtual environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "  Run: install.ps1 to set up the environment" -ForegroundColor Yellow
    exit 1
}

# Test 2: Check Ollama
Write-Host "`n[2/5] Checking Ollama..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
    Write-Host "  [OK] Ollama is running" -ForegroundColor Green

    $models = $response.models | ForEach-Object { $_.name }
    if ($models -contains "nomic-embed-text:latest") {
        Write-Host "  [OK] Embedding model available" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Embedding model not found. Run: ollama pull nomic-embed-text" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] Ollama is not running. Start with: ollama serve" -ForegroundColor Red
    exit 1
}

# Test 3: Check imports
Write-Host "`n[3/5] Checking Python imports..." -ForegroundColor Yellow
$testImports = @"
import sys
sys.path.insert(0, r'$ProjectRoot')
try:
    import chromadb
    import ollama
    import mcp
    import pymupdf
    from mcp_server.server import KnowledgeOrchestrator
    print("OK")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"@

$result = & $VenvPython -c $testImports 2>&1
if ($result -eq "OK") {
    Write-Host "  [OK] All imports successful" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Import failed: $result" -ForegroundColor Red
    exit 1
}

# Test 4: Check ChromaDB
Write-Host "`n[4/5] Checking ChromaDB connection..." -ForegroundColor Yellow
$testChroma = @"
import sys
sys.path.insert(0, r'$ProjectRoot')
from mcp_server.server import KnowledgeOrchestrator
orch = KnowledgeOrchestrator()
count = orch.collection.count()
print(f"OK:{count}")
"@

$result = & $VenvPython -c $testChroma 2>&1 | Select-String "OK:" | ForEach-Object { $_.ToString() }
if ($result -match "OK:(\d+)") {
    $chunks = $Matches[1]
    Write-Host "  [OK] ChromaDB connected - $chunks chunks indexed" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] ChromaDB connection failed" -ForegroundColor Red
    exit 1
}

# Test 5: Test search
Write-Host "`n[5/5] Testing semantic search..." -ForegroundColor Yellow
$testSearch = @"
import sys
sys.path.insert(0, r'$ProjectRoot')
from mcp_server.server import KnowledgeOrchestrator
orch = KnowledgeOrchestrator()
results = orch.query("test query", max_results=1)
print(f"OK:{len(results)}")
"@

$result = & $VenvPython -c $testSearch 2>&1 | Select-String "OK:" | ForEach-Object { $_.ToString() }
if ($result -match "OK:(\d+)") {
    $resultCount = $Matches[1]
    if ($resultCount -gt 0) {
        Write-Host "  [OK] Search returned $resultCount result(s)" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Search returned no results (index may be empty)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [ERROR] Search failed" -ForegroundColor Red
}

Write-Host "`n=== ALL TESTS PASSED ===" -ForegroundColor Green
Write-Host "Knowledge RAG system is ready to use!`n" -ForegroundColor Cyan
