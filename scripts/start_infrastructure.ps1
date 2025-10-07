# ============================================================================
# A.D.R.I.A.N Infrastructure Check Script (Windows PowerShell)
# Verifies Redis and PostgreSQL services are running (Native Windows)
# ============================================================================

Write-Host "Checking ADRIAN Infrastructure Services..." -ForegroundColor Cyan

# Check Redis service
Write-Host "`nChecking Redis..." -ForegroundColor Yellow
$redisService = Get-Service -Name redis-server -ErrorAction SilentlyContinue

if ($redisService) {
    if ($redisService.Status -eq 'Running') {
        Write-Host "✓ Redis is running" -ForegroundColor Green
    } else {
        Write-Host "⚠ Redis service exists but is not running" -ForegroundColor Yellow
        Write-Host "  Starting Redis..." -ForegroundColor Gray
        Start-Service -Name redis-server
        Write-Host "✓ Redis started" -ForegroundColor Green
    }
} else {
    Write-Host "✗ Redis service not found" -ForegroundColor Red
    Write-Host "  Run: .\scripts\install_infrastructure.ps1 (as Administrator)" -ForegroundColor Yellow
    exit 1
}

# Check PostgreSQL service
Write-Host "`nChecking PostgreSQL..." -ForegroundColor Yellow
$pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($pgService) {
    if ($pgService.Status -eq 'Running') {
        Write-Host "✓ PostgreSQL is running" -ForegroundColor Green
    } else {
        Write-Host "⚠ PostgreSQL service exists but is not running" -ForegroundColor Yellow
        Write-Host "  Starting PostgreSQL..." -ForegroundColor Gray
        Start-Service -Name $pgService.Name
        Write-Host "✓ PostgreSQL started" -ForegroundColor Green
    }
} else {
    Write-Host "✗ PostgreSQL service not found" -ForegroundColor Red
    Write-Host "  Run: .\scripts\install_infrastructure.ps1 (as Administrator)" -ForegroundColor Yellow
    exit 1
}

# Test connectivity
Write-Host "`nTesting connectivity..." -ForegroundColor Yellow

# Test Redis
try {
    $redisTest = Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($redisTest) {
        Write-Host "✓ Redis is accessible on port 6379" -ForegroundColor Green
    } else {
        Write-Host "⚠ Redis port 6379 not accessible" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Could not test Redis connectivity" -ForegroundColor Yellow
}

# Test PostgreSQL
try {
    $pgTest = Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($pgTest) {
        Write-Host "✓ PostgreSQL is accessible on port 5432" -ForegroundColor Green
    } else {
        Write-Host "⚠ PostgreSQL port 5432 not accessible" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Could not test PostgreSQL connectivity" -ForegroundColor Yellow
}

Write-Host "`n✓ Infrastructure check complete!" -ForegroundColor Green
Write-Host "  Redis: localhost:6379" -ForegroundColor Cyan
Write-Host "  PostgreSQL: localhost:5432" -ForegroundColor Cyan

