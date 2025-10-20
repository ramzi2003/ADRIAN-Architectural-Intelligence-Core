# ============================================================================
# Install Audio Dependencies for Phase 2
# ============================================================================

Write-Host "Installing ADRIAN Audio Dependencies..." -ForegroundColor Cyan

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "X Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "OK Virtual environment found" -ForegroundColor Green

# Install audio packages
Write-Host "`nInstalling audio libraries..." -ForegroundColor Yellow
Write-Host "  - sounddevice (microphone/speaker)" -ForegroundColor Gray
Write-Host "  - soundfile (audio file I/O)" -ForegroundColor Gray
Write-Host "  - Coqui TTS (text-to-speech)" -ForegroundColor Gray

& "venv\Scripts\pip.exe" install sounddevice soundfile TTS

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nOK Audio dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "`nX Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Create test directory
if (-not (Test-Path "test_audio")) {
    New-Item -ItemType Directory -Path "test_audio" | Out-Null
    Write-Host "OK Created test_audio directory" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "OK Installation Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test audio: python services\io_service\audio_test.py" -ForegroundColor Gray
Write-Host "  2. Make sure microphone and speakers are ready" -ForegroundColor Gray
Write-Host "  3. Run the test and speak when prompted" -ForegroundColor Gray

