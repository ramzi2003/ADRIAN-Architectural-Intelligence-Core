# ============================================================================
# A.D.R.I.A.N Database Initialization Script
# Runs SQL schema creation and FAISS index initialization
# ============================================================================

Write-Host "Initializing ADRIAN Database..." -ForegroundColor Cyan

# Check PostgreSQL is running
$pgService = Get-Service postgresql* -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $pgService -or $pgService.Status -ne 'Running') {
    Write-Host "X PostgreSQL is not running!" -ForegroundColor Red
    exit 1
}

Write-Host "OK PostgreSQL is running" -ForegroundColor Green

# ============================================================================
# Step 1: Run SQL Schema Creation
# ============================================================================

Write-Host "`nStep 1: Creating database tables..." -ForegroundColor Yellow

$pgPath = "C:\Program Files\PostgreSQL\17\bin"
$sqlFile = "database\init_database.sql"

if (-not (Test-Path $sqlFile)) {
    Write-Host "X SQL file not found: $sqlFile" -ForegroundColor Red
    exit 1
}

# Run SQL script
& "$pgPath\psql.exe" -U adrian -d adrian -f $sqlFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK Database tables created successfully" -ForegroundColor Green
} else {
    Write-Host "X Failed to create database tables" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Step 2: Verify Tables
# ============================================================================

Write-Host "`nStep 2: Verifying tables..." -ForegroundColor Yellow

$tables = & "$pgPath\psql.exe" -U adrian -d adrian -t -c "\dt" 2>$null

if ($tables -match "users" -and $tables -match "conversations" -and $tables -match "memory_embeddings") {
    Write-Host "OK All tables created" -ForegroundColor Green
    Write-Host "Tables:" -ForegroundColor Cyan
    Write-Host $tables -ForegroundColor Gray
} else {
    Write-Host "! Some tables may be missing" -ForegroundColor Yellow
}

# ============================================================================
# Step 3: Initialize FAISS Index
# ============================================================================

Write-Host "`nStep 3: Initializing FAISS index..." -ForegroundColor Yellow

# Set PYTHONPATH
$env:PYTHONPATH = $PWD

# Run Python initialization script
& "venv\Scripts\python.exe" "database\init_faiss_index.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK FAISS index initialized" -ForegroundColor Green
} else {
    Write-Host "X Failed to initialize FAISS index" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Summary
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "OK Database Setup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Created:" -ForegroundColor Yellow
Write-Host "  * PostgreSQL tables (users, conversations, memory_embeddings, etc.)" -ForegroundColor Gray
Write-Host "  * FAISS vector index" -ForegroundColor Gray
Write-Host "  * Database views" -ForegroundColor Gray
Write-Host "  * Default user account" -ForegroundColor Gray

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Memory service will now use real storage" -ForegroundColor Gray
Write-Host "  2. Restart services: .\scripts\start_services.ps1" -ForegroundColor Gray
Write-Host "  3. Test memory storage and retrieval" -ForegroundColor Gray

