# ============================================================================
# A.D.R.I.A.N Service Health Check Script
# Tests all service health endpoints
# ============================================================================

Write-Host "Testing ADRIAN Services Health..." -ForegroundColor Cyan

$services = @(
    @{Name="IO Service"; Port=8001},
    @{Name="Processing Service"; Port=8002},
    @{Name="Memory Service"; Port=8003},
    @{Name="Execution Service"; Port=8004},
    @{Name="Security Service"; Port=8005},
    @{Name="Output Service"; Port=8006}
)

$all_healthy = $true

foreach ($service in $services) {
    $url = "http://localhost:$($service.Port)/health"
    
    try {
        $response = Invoke-RestMethod -Uri $url -TimeoutSec 5
        if ($response.status -eq "healthy") {
            Write-Host "OK $($service.Name) is healthy" -ForegroundColor Green
        } else {
            Write-Host "! $($service.Name) is unhealthy" -ForegroundColor Yellow
            $all_healthy = $false
        }
    } catch {
        Write-Host "X $($service.Name) is unreachable" -ForegroundColor Red
        $all_healthy = $false
    }
}

if ($all_healthy) {
    Write-Host "`nOK All services are healthy!" -ForegroundColor Green
} else {
    Write-Host "`n! Some services are not healthy" -ForegroundColor Yellow
}
