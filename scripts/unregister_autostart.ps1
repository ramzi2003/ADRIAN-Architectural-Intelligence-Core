# ============================================================================
# A.D.R.I.A.N Auto-Start Removal Script
# Removes Task Scheduler task for ADRIAN auto-start
# ============================================================================

Write-Host "Removing ADRIAN Auto-Start..." -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "✗ This script requires Administrator privileges." -ForegroundColor Red
    Write-Host "  Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

$taskName = "ADRIAN-AutoStart"

# Remove Task Scheduler task
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "✓ Removed Task Scheduler task: $taskName" -ForegroundColor Green
} else {
    Write-Host "⚠ Task '$taskName' not found" -ForegroundColor Yellow
}

Write-Host "`n✓ Auto-start removed. Services will not start automatically on login." -ForegroundColor Green

