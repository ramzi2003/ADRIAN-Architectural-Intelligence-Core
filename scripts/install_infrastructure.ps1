# ============================================================================
# A.D.R.I.A.N Infrastructure Installation Script (Native Windows)
# Installs Redis and PostgreSQL as Windows services (auto-start)
# ============================================================================

Write-Host "Installing ADRIAN Infrastructure (Native Windows)..." -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "X This script requires Administrator privileges." -ForegroundColor Red
    Write-Host "  Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "OK Running with Administrator privileges" -ForegroundColor Green

# ============================================================================
# Install Chocolatey (Windows Package Manager)
# ============================================================================

Write-Host "`nChecking Chocolatey..." -ForegroundColor Yellow
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Host "OK Chocolatey installed" -ForegroundColor Green
} else {
    Write-Host "OK Chocolatey already installed" -ForegroundColor Green
}

# ============================================================================
# Install Redis
# ============================================================================

Write-Host "`nInstalling Redis..." -ForegroundColor Yellow
if (-not (Get-Service redis-server -ErrorAction SilentlyContinue)) {
    # Download and install Redis for Windows
    $redisVersion = "5.0.14.1"
    $redisUrl = "https://github.com/microsoftarchive/redis/releases/download/win-$redisVersion/Redis-x64-$redisVersion.msi"
    $redisInstaller = "$env:TEMP\redis-installer.msi"
    
    Write-Host "  Downloading Redis..." -ForegroundColor Gray
    Invoke-WebRequest -Uri $redisUrl -OutFile $redisInstaller
    
    Write-Host "  Installing Redis..." -ForegroundColor Gray
    Start-Process msiexec.exe -ArgumentList "/i", $redisInstaller, "/quiet", "/norestart" -Wait
    
    # Wait for service to be created
    Start-Sleep -Seconds 5
    
    # Set Redis service to auto-start
    Set-Service -Name redis-server -StartupType Automatic
    Start-Service -Name redis-server
    
    Write-Host "OK Redis installed and started" -ForegroundColor Green
    Write-Host "  Service: redis-server (auto-start enabled)" -ForegroundColor Gray
} else {
    Write-Host "OK Redis already installed" -ForegroundColor Green
    Set-Service -Name redis-server -StartupType Automatic
    if ((Get-Service redis-server).Status -ne 'Running') {
        Start-Service -Name redis-server
        Write-Host "  Started Redis service" -ForegroundColor Gray
    }
}

# Test Redis connection
try {
    $null = Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet -WarningAction SilentlyContinue
    Write-Host "OK Redis is accessible on port 6379" -ForegroundColor Green
} catch {
    Write-Host "! Redis may not be ready yet" -ForegroundColor Yellow
}

# ============================================================================
# Install PostgreSQL
# ============================================================================

Write-Host "`nInstalling PostgreSQL..." -ForegroundColor Yellow
if (-not (Get-Service postgresql* -ErrorAction SilentlyContinue)) {
    choco install postgresql15 --params '/Password:adrian_password' -y
    
    # Wait for installation to complete
    Start-Sleep -Seconds 10
    
    # Find PostgreSQL service (version-specific name)
    $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1
    
    if ($pgService) {
        Set-Service -Name $pgService.Name -StartupType Automatic
        if ($pgService.Status -ne 'Running') {
            Start-Service -Name $pgService.Name
        }
        Write-Host "OK PostgreSQL installed and started" -ForegroundColor Green
        Write-Host "  Service: $($pgService.Name) (auto-start enabled)" -ForegroundColor Gray
    } else {
        Write-Host "! PostgreSQL service not found. Manual configuration may be needed." -ForegroundColor Yellow
    }
} else {
    Write-Host "OK PostgreSQL already installed" -ForegroundColor Green
    $pgService = Get-Service -Name "postgresql*" | Select-Object -First 1
    Set-Service -Name $pgService.Name -StartupType Automatic
    if ($pgService.Status -ne 'Running') {
        Start-Service -Name $pgService.Name
        Write-Host "  Started PostgreSQL service" -ForegroundColor Gray
    }
}

# Test PostgreSQL connection
Start-Sleep -Seconds 3
try {
    $null = Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet -WarningAction SilentlyContinue
    Write-Host "OK PostgreSQL is accessible on port 5432" -ForegroundColor Green
} catch {
    Write-Host "! PostgreSQL may not be ready yet" -ForegroundColor Yellow
}

# ============================================================================
# Create PostgreSQL Database
# ============================================================================

Write-Host "`nConfiguring PostgreSQL database..." -ForegroundColor Yellow
$pgPath = "C:\Program Files\PostgreSQL\15\bin"
if (Test-Path $pgPath) {
    $env:Path += ";$pgPath"
    
    Write-Host "  Creating database user and database..." -ForegroundColor Gray
    
    # Create user (ignore error if already exists)
    & "$pgPath\psql.exe" -U postgres -h localhost -p 5432 -c "CREATE USER adrian WITH PASSWORD 'adrian_password';" 2>$null
    
    # Create database (ignore error if already exists)
    & "$pgPath\psql.exe" -U postgres -h localhost -p 5432 -c "CREATE DATABASE adrian OWNER adrian;" 2>$null
    
    # Grant privileges
    & "$pgPath\psql.exe" -U postgres -h localhost -p 5432 -c "GRANT ALL PRIVILEGES ON DATABASE adrian TO adrian;" 2>$null
    
    Write-Host "OK Database 'adrian' created/configured" -ForegroundColor Green
} else {
    Write-Host "! PostgreSQL not found at expected location" -ForegroundColor Yellow
    Write-Host "  You may need to manually create the database after installation completes" -ForegroundColor Gray
}

# ============================================================================
# Summary
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "OK Infrastructure Installation Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Installed Services (Auto-start on boot):" -ForegroundColor Yellow
Write-Host "  * Redis: localhost:6379" -ForegroundColor Gray
Write-Host "  * PostgreSQL: localhost:5432" -ForegroundColor Gray
Write-Host "  * Database: adrian" -ForegroundColor Gray
Write-Host "  * User: adrian / adrian_password" -ForegroundColor Gray

Write-Host "`nService Status:" -ForegroundColor Yellow
$redisStatus = (Get-Service redis-server -ErrorAction SilentlyContinue).Status
$pgStatus = (Get-Service postgresql* -ErrorAction SilentlyContinue | Select-Object -First 1).Status
Write-Host "  Redis: $redisStatus" -ForegroundColor $(if ($redisStatus -eq 'Running') {'Green'} else {'Red'})
Write-Host "  PostgreSQL: $pgStatus" -ForegroundColor $(if ($pgStatus -eq 'Running') {'Green'} else {'Red'})

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Services will auto-start on system boot" -ForegroundColor Gray
Write-Host "  2. Run: .\scripts\setup.ps1 (if not done)" -ForegroundColor Gray
Write-Host "  3. Run: .\scripts\register_autostart.ps1" -ForegroundColor Gray
Write-Host "  4. Run: .\scripts\start_services.ps1" -ForegroundColor Gray
