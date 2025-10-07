# ============================================================================
# A.D.R.I.A.N Services Stop Script (Windows PowerShell)
# Stops all ADRIAN microservices
# ============================================================================

Write-Host "Stopping ADRIAN Services..." -ForegroundColor Cyan

# Kill all Python processes running ADRIAN services
Get-Process python -ErrorAction SilentlyContinue | 
    Where-Object {$_.CommandLine -like "*services*main.py*"} | 
    ForEach-Object {
        Write-Host "Stopping PID $($_.Id)..." -ForegroundColor Yellow
        Stop-Process -Id $_.Id -Force
    }

Write-Host "OK All services stopped!" -ForegroundColor Green
