# ============================================================================
# A.D.R.I.A.N Services Startup Script (Windows PowerShell)
# Starts all ADRIAN microservices in separate PowerShell windows
# ============================================================================

Write-Host "Starting ADRIAN Services..." -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "X Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment path
$venv_python = "venv\Scripts\python.exe"

# Array of services
$services = @(
    @{Name="IO Service"; Script="services\io_service\main.py"; Port=8001},
    @{Name="Processing Service"; Script="services\processing_service\main.py"; Port=8002},
    @{Name="Memory Service"; Script="services\memory_service\main.py"; Port=8003},
    @{Name="Execution Service"; Script="services\execution_service\main.py"; Port=8004},
    @{Name="Security Service"; Script="services\security_service\main.py"; Port=8005},
    @{Name="Output Service"; Script="services\output_service\main.py"; Port=8006}
)

# Get the project root directory
$projectRoot = Get-Location

# Start each service in a new PowerShell window
foreach ($service in $services) {
    Write-Host "Starting $($service.Name) on port $($service.Port)..." -ForegroundColor Yellow
    
    $title = $service.Name
    $script = $service.Script
    
    # Set PYTHONPATH to include project root so 'shared' module can be found
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { `$env:PYTHONPATH='$projectRoot'; Set-Location '$projectRoot'; `$host.UI.RawUI.WindowTitle = '$title'; & '$venv_python' '$script' }"
    
    Start-Sleep -Milliseconds 500
}

Write-Host "`nOK All services started!" -ForegroundColor Green
Write-Host "`nService URLs:" -ForegroundColor Cyan
foreach ($service in $services) {
    Write-Host "  $($service.Name): http://localhost:$($service.Port)/health" -ForegroundColor Gray
}

Write-Host "`nServices are running in separate windows." -ForegroundColor Yellow
Write-Host "Close those windows to stop the services." -ForegroundColor Yellow
