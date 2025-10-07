# ============================================================================
# A.D.R.I.A.N Setup Script (Windows PowerShell)
# Sets up Python virtual environment and installs dependencies
# ============================================================================

Write-Host "Setting up ADRIAN Development Environment..." -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
try {
    $python_version = python --version
    Write-Host "OK Found: $python_version" -ForegroundColor Green
} catch {
    Write-Host "X Python not found. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "OK Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "`nOK Virtual environment already exists" -ForegroundColor Green
}

# Activate and install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
& "venv\Scripts\pip.exe" install --upgrade pip
& "venv\Scripts\pip.exe" install -r requirements.txt

Write-Host "`nOK Dependencies installed!" -ForegroundColor Green

# Create necessary directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
$directories = @("logs", "data", "data/faiss_index")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}

# Copy environment template
if (-not (Test-Path ".env")) {
    Write-Host "`nCreating .env file from template..." -ForegroundColor Yellow
    Copy-Item "env.example" ".env"
    Write-Host "OK .env file created. Please update with your settings." -ForegroundColor Green
} else {
    Write-Host "`nOK .env file already exists" -ForegroundColor Green
}

Write-Host "`nOK Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env file with your configuration" -ForegroundColor Gray
Write-Host "  2. Run: .\scripts\start_infrastructure.ps1" -ForegroundColor Gray
Write-Host "  3. Install Ollama and pull mistral:7b model" -ForegroundColor Gray
Write-Host "  4. Run: .\scripts\start_services.ps1" -ForegroundColor Gray
