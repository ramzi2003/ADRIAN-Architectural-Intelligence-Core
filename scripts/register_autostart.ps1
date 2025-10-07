# ============================================================================
# A.D.R.I.A.N Auto-Start Registration Script
# Creates Windows Task Scheduler tasks to auto-start ADRIAN services on login
# ============================================================================

Write-Host "Registering ADRIAN for Auto-Start..." -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "✗ This script requires Administrator privileges." -ForegroundColor Red
    Write-Host "  Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Get project root directory
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$startScript = Join-Path $projectRoot "scripts\start_services_background.ps1"

# Verify virtual environment exists
if (-not (Test-Path $pythonExe)) {
    Write-Host "✗ Virtual environment not found." -ForegroundColor Red
    Write-Host "  Run .\scripts\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment found" -ForegroundColor Green

# ============================================================================
# Create background startup script
# ============================================================================

Write-Host "`nCreating background startup script..." -ForegroundColor Yellow

$startupScriptContent = @"
# ADRIAN Background Startup Script
# Starts all services in hidden windows

`$projectRoot = "$projectRoot"
`$venvPython = "$pythonExe"

# Array of services
`$services = @(
    @{Name="io-service"; Script="services\io_service\main.py"; Port=8001},
    @{Name="processing-service"; Script="services\processing_service\main.py"; Port=8002},
    @{Name="memory-service"; Script="services\memory_service\main.py"; Port=8003},
    @{Name="execution-service"; Script="services\execution_service\main.py"; Port=8004},
    @{Name="security-service"; Script="services\security_service\main.py"; Port=8005},
    @{Name="output-service"; Script="services\output_service\main.py"; Port=8006}
)

Set-Location `$projectRoot

foreach (`$service in `$services) {
    `$scriptPath = Join-Path `$projectRoot `$service.Script
    
    # Start service in hidden window
    Start-Process -FilePath `$venvPython -ArgumentList `$scriptPath -WindowStyle Hidden
    
    Start-Sleep -Milliseconds 500
}

# Log startup
`$logPath = Join-Path `$projectRoot "logs\autostart.log"
`$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path `$logPath -Value "`$timestamp - ADRIAN services started"
"@

$startupScriptPath = Join-Path $projectRoot "scripts\start_services_background.ps1"
Set-Content -Path $startupScriptPath -Value $startupScriptContent
Write-Host "✓ Background startup script created" -ForegroundColor Green

# ============================================================================
# Create Task Scheduler Task
# ============================================================================

Write-Host "`nRegistering Task Scheduler task..." -ForegroundColor Yellow

$taskName = "ADRIAN-AutoStart"
$taskDescription = "Automatically start ADRIAN AI Assistant services on login"

# Remove existing task if present
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "  Removed existing task" -ForegroundColor Gray
}

# Create new task
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" `
    -Argument "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$startupScriptPath`""

$trigger = New-ScheduledTaskTrigger -AtLogon -User $env:USERNAME

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

Register-ScheduledTask `
    -TaskName $taskName `
    -Description $taskDescription `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal | Out-Null

Write-Host "✓ Task Scheduler task registered: $taskName" -ForegroundColor Green

# ============================================================================
# Create manual stop script
# ============================================================================

Write-Host "`nCreating manual stop script..." -ForegroundColor Yellow

$stopScriptContent = @"
# Stop all ADRIAN services
Write-Host "Stopping ADRIAN services..." -ForegroundColor Yellow

Get-Process python -ErrorAction SilentlyContinue | 
    Where-Object {`$_.CommandLine -like "*services*main.py*"} | 
    ForEach-Object {
        Stop-Process -Id `$_.Id -Force
    }

Write-Host "✓ All ADRIAN services stopped" -ForegroundColor Green
"@

$stopScriptPath = Join-Path $projectRoot "scripts\stop_services_background.ps1"
Set-Content -Path $stopScriptPath -Value $stopScriptContent
Write-Host "✓ Stop script created" -ForegroundColor Green

# ============================================================================
# Summary
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✓ Auto-Start Registration Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  • Task Name: $taskName" -ForegroundColor Gray
Write-Host "  • Trigger: At user login ($env:USERNAME)" -ForegroundColor Gray
Write-Host "  • Services: 6 microservices" -ForegroundColor Gray
Write-Host "  • Mode: Background (hidden windows)" -ForegroundColor Gray

Write-Host "`nFiles Created:" -ForegroundColor Yellow
Write-Host "  • $startupScriptPath" -ForegroundColor Gray
Write-Host "  • $stopScriptPath" -ForegroundColor Gray

Write-Host "`nManual Controls:" -ForegroundColor Yellow
Write-Host "  • Start services: .\scripts\start_services.ps1" -ForegroundColor Gray
Write-Host "  • Stop services: .\scripts\stop_services_background.ps1" -ForegroundColor Gray
Write-Host "  • View in Task Scheduler: taskschd.msc" -ForegroundColor Gray

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  • Services will auto-start on next login" -ForegroundColor Gray
Write-Host "  • To test now: Run .\scripts\start_services.ps1" -ForegroundColor Gray
Write-Host "  • To disable: Open Task Scheduler and disable '$taskName'" -ForegroundColor Gray

