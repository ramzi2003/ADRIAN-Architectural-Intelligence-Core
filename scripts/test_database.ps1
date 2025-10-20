# ============================================================================
# A.D.R.I.A.N Database Test Script
# Tests PostgreSQL tables and FAISS index
# ============================================================================

Write-Host "Testing ADRIAN Database Setup..." -ForegroundColor Cyan

# Set PYTHONPATH
$env:PYTHONPATH = $PWD

# ============================================================================
# Test 1: PostgreSQL Tables
# ============================================================================

Write-Host "`nTest 1: Checking PostgreSQL tables..." -ForegroundColor Yellow

$pgPath = "C:\Program Files\PostgreSQL\17\bin"
$tables = & "$pgPath\psql.exe" -U adrian -d adrian -t -c "\dt" 2>$null

$requiredTables = @("users", "sessions", "conversations", "memory_embeddings", "preferences", "tasks", "system_logs")
$allFound = $true

foreach ($table in $requiredTables) {
    if ($tables -match $table) {
        Write-Host "  OK Found table: $table" -ForegroundColor Green
    } else {
        Write-Host "  X Missing table: $table" -ForegroundColor Red
        $allFound = $false
    }
}

if (-not $allFound) {
    Write-Host "`nX Some tables are missing. Run: .\scripts\init_database.ps1" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Test 2: Default User
# ============================================================================

Write-Host "`nTest 2: Checking default user..." -ForegroundColor Yellow

$userCheck = & "$pgPath\psql.exe" -U adrian -d adrian -t -c "SELECT username FROM users WHERE username='default_user';" 2>$null

if ($userCheck -match "default_user") {
    Write-Host "  OK Default user exists" -ForegroundColor Green
} else {
    Write-Host "  ! Default user not found" -ForegroundColor Yellow
}

# ============================================================================
# Test 3: FAISS Index Files
# ============================================================================

Write-Host "`nTest 3: Checking FAISS index files..." -ForegroundColor Yellow

if (Test-Path "data\faiss_index\index.faiss") {
    Write-Host "  OK FAISS index file exists" -ForegroundColor Green
} else {
    Write-Host "  X FAISS index file not found" -ForegroundColor Red
    Write-Host "    Run: .\scripts\init_database.ps1" -ForegroundColor Yellow
    $allFound = $false
}

if (Test-Path "data\faiss_index\metadata.json") {
    Write-Host "  OK FAISS metadata file exists" -ForegroundColor Green
} else {
    Write-Host "  X FAISS metadata file not found" -ForegroundColor Red
    $allFound = $false
}

# ============================================================================
# Test 4: Memory Service API
# ============================================================================

Write-Host "`nTest 4: Testing Memory Service API..." -ForegroundColor Yellow

try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8003/health" -TimeoutSec 5
    
    Write-Host "  OK Memory Service is healthy" -ForegroundColor Green
    Write-Host "    PostgreSQL: $($healthResponse.dependencies.postgresql)" -ForegroundColor Gray
    Write-Host "    FAISS: $($healthResponse.dependencies.faiss)" -ForegroundColor Gray
    Write-Host "    FAISS Vectors: $($healthResponse.dependencies.faiss_vectors)" -ForegroundColor Gray
    
} catch {
    Write-Host "  X Memory Service is not running" -ForegroundColor Red
    Write-Host "    Start services first: .\scripts\start_services.ps1" -ForegroundColor Yellow
}

# ============================================================================
# Test 5: Store and Search Memory
# ============================================================================

Write-Host "`nTest 5: Testing memory storage and search..." -ForegroundColor Yellow

try {
    # Store a test memory
    $storeBody = @{
        text = "ADRIAN is a sophisticated AI assistant built with microservices architecture"
        user_id = "test_user"
        metadata = @{
            source_type = "test"
            test = $true
        }
    } | ConvertTo-Json
    
    $storeResult = Invoke-RestMethod -Uri "http://localhost:8003/memory/store" `
        -Method POST -Body $storeBody -ContentType "application/json"
    
    Write-Host "  OK Memory stored: $($storeResult.memory_id)" -ForegroundColor Green
    
    # Wait a moment for indexing
    Start-Sleep -Seconds 1
    
    # Search for similar memory
    $searchBody = @{
        query = "What is ADRIAN?"
        limit = 3
        user_id = "test_user"
    } | ConvertTo-Json
    
    $searchResult = Invoke-RestMethod -Uri "http://localhost:8003/memory/search" `
        -Method POST -Body $searchBody -ContentType "application/json"
    
    if ($searchResult.results.Count -gt 0) {
        Write-Host "  OK Search returned $($searchResult.results.Count) result(s)" -ForegroundColor Green
        Write-Host "    Top result: $($searchResult.results[0].text.Substring(0, [Math]::Min(50, $searchResult.results[0].text.Length)))..." -ForegroundColor Gray
    } else {
        Write-Host "  ! Search returned no results (index may be empty)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "  X Memory API test failed: $($_.Exception.Message)" -ForegroundColor Red
}

# ============================================================================
# Summary
# ============================================================================

if ($allFound) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "OK Database Setup Verified!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
} else {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "! Some tests failed - see above" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
}

