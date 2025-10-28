# PowerShell script to install espeak-ng automatically
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Installing espeak-ng for Coqui TTS (Male British Voice)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host ""
    Write-Host "ERROR: This script needs Administrator privileges" -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    Write-Host "Then run: .\install_espeak.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 1: Downloading espeak-ng..." -ForegroundColor Green

# Download espeak-ng installer
$downloadUrl = "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi"
$installerPath = "$env:TEMP\espeak-ng-X64.msi"

try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "   Downloaded successfully" -ForegroundColor Green
} catch {
    Write-Host "   ERROR: Failed to download espeak-ng" -ForegroundColor Red
    Write-Host "   Please download manually from: $downloadUrl" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 2: Installing espeak-ng..." -ForegroundColor Green

# Install silently
try {
    Start-Process msiexec.exe -ArgumentList "/i `"$installerPath`" /quiet /norestart" -Wait -NoNewWindow
    Write-Host "   Installed successfully" -ForegroundColor Green
} catch {
    Write-Host "   ERROR: Installation failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 3: Adding to PATH..." -ForegroundColor Green

# Add to system PATH
$espeakPath = "C:\Program Files\eSpeak NG"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

if ($currentPath -notlike "*$espeakPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$espeakPath", "Machine")
    Write-Host "   Added to PATH successfully" -ForegroundColor Green
} else {
    Write-Host "   Already in PATH" -ForegroundColor Yellow
}

# Update current session PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host ""
Write-Host "Step 4: Verifying installation..." -ForegroundColor Green

# Test espeak-ng
try {
    $espeakExe = "C:\Program Files\eSpeak NG\espeak-ng.exe"
    if (Test-Path $espeakExe) {
        Write-Host "   espeak-ng found at: $espeakExe" -ForegroundColor Green
        
        # Test it
        & $espeakExe --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   espeak-ng is working!" -ForegroundColor Green
        }
    } else {
        Write-Host "   WARNING: espeak-ng.exe not found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   WARNING: Could not verify espeak-ng" -ForegroundColor Yellow
}

# Cleanup
Remove-Item $installerPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Close and reopen your terminal (to refresh PATH)" -ForegroundColor White
Write-Host "2. Run: python services/io_service/test_tts.py --test basic" -ForegroundColor White
Write-Host ""
Write-Host "You will now have male British voice (JARVIS-like)!" -ForegroundColor Green
Write-Host ""

