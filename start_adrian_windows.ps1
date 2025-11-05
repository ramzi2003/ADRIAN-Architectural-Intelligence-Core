# ADRIAN Windows Startup Script
# Starts IO Service in Windows (Processing + Output Services run separately in WSL)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[*] Starting ADRIAN IO Service in Windows" -ForegroundColor Cyan
Write-Host "   (Voice Input, STT, Hotword Detection)" -ForegroundColor Cyan
Write-Host "   Note: Processing + Output Services should run in WSL" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Redis connection
Write-Host "[*] Checking Redis connection..." -ForegroundColor Yellow
try {
    $redisTest = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue
    if (-not $redisTest.TcpTestSucceeded) {
        Write-Host "[X] Redis is not running on localhost:6379" -ForegroundColor Red
        Write-Host "   Please start Redis in WSL: sudo service redis-server start" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[+] Redis is running" -ForegroundColor Green
} catch {
    Write-Host "[X] Cannot check Redis connection" -ForegroundColor Red
    Write-Host "   Make sure Redis is running in WSL" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "services\io_service\main.py")) {
    Write-Host "[X] Please run this script from the ADRIAN project root directory" -ForegroundColor Red
    exit 1
}

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[!] Virtual environment not detected. Checking for venv..." -ForegroundColor Yellow
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Host "   Activating venv..." -ForegroundColor Yellow
        & "venv\Scripts\Activate.ps1"
    } else {
        Write-Host "[X] Virtual environment not found. Please create venv first" -ForegroundColor Red
        Write-Host "   python -m venv venv" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "[*] Starting IO Service..." -ForegroundColor Cyan
Write-Host "   This service needs Windows microphone access" -ForegroundColor Yellow
Write-Host ""

# Start IO Service
python services\io_service\main.py

Write-Host ""
Write-Host "[!] IO Service stopped" -ForegroundColor Yellow


