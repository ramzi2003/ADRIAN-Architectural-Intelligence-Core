# Infrastructure Setup - Native Windows Services

## Overview

ADRIAN uses **native Windows services** for Redis and PostgreSQL. No Docker required!

✅ **Advantages**:
- Auto-start on Windows boot
- Lower resource usage (no Docker overhead)
- Faster startup times
- Native Windows integration
- No need to manually start infrastructure

⚠️ **Requirements**:
- Administrator privileges for initial installation
- Services are installed system-wide

---

## What Gets Installed

### Redis
- **Location**: `C:\Program Files\Redis`
- **Service Name**: `redis-server`
- **Port**: 6379
- **Auto-Start**: Yes (Automatic)
- **RAM Usage**: ~10MB idle

### PostgreSQL 15
- **Location**: `C:\Program Files\PostgreSQL\15`
- **Service Name**: `postgresql-x64-15` (version-dependent)
- **Port**: 5432
- **Auto-Start**: Yes (Automatic)
- **RAM Usage**: ~50MB idle
- **Database**: `adrian`
- **User**: `adrian`
- **Password**: `adrian_password`

---

## Installation

### One-Time Setup (Requires Administrator)

```powershell
# Run PowerShell as Administrator
.\scripts\install_infrastructure.ps1
```

This script will:
1. Install Chocolatey (if not present)
2. Download and install Redis for Windows
3. Install PostgreSQL via Chocolatey
4. Create `adrian` database and user
5. Configure both services to auto-start on boot
6. Start both services immediately

---

## Managing Services

### Check Status

```powershell
# Check if services are running
Get-Service redis-server
Get-Service postgresql*
```

### Start/Stop Services

```powershell
# Start services
Start-Service redis-server
Start-Service postgresql-x64-15

# Stop services
Stop-Service redis-server
Stop-Service postgresql-x64-15
```

### Disable/Enable Auto-Start

```powershell
# Disable auto-start
Set-Service redis-server -StartupType Manual
Set-Service postgresql-x64-15 -StartupType Manual

# Re-enable auto-start
Set-Service redis-server -StartupType Automatic
Set-Service postgresql-x64-15 -StartupType Automatic
```

### Test Connectivity

```powershell
# Test if services are accessible
Test-NetConnection -ComputerName localhost -Port 6379  # Redis
Test-NetConnection -ComputerName localhost -Port 5432  # PostgreSQL
```

---

## Uninstallation

### Remove Redis

1. Stop the service:
   ```powershell
   Stop-Service redis-server
   ```
2. Control Panel → Programs and Features → Uninstall Redis

### Remove PostgreSQL

```powershell
# Using Chocolatey
choco uninstall postgresql15 -y
```

Or use Windows "Add or Remove Programs"

---

## Troubleshooting

### Service Won't Start

**Check Windows Event Viewer:**
```powershell
Get-EventLog -LogName Application -Source *redis* -Newest 10
Get-EventLog -LogName Application -Source *postgresql* -Newest 10
```

**Verify installation:**
```powershell
# Redis
Test-Path "C:\Program Files\Redis\redis-server.exe"

# PostgreSQL
Test-Path "C:\Program Files\PostgreSQL\15\bin\postgres.exe"
```

### Port Already in Use

**Check what's using the port:**
```powershell
# Check port 6379 (Redis)
netstat -ano | findstr :6379

# Check port 5432 (PostgreSQL)
netstat -ano | findstr :5432
```

**Kill the process if needed:**
```powershell
Stop-Process -Id <PID> -Force
```

### Permission Denied

- Ensure you're running PowerShell as Administrator
- Disable antivirus temporarily during installation
- Check Windows User Account Control (UAC) settings

---

## Security

### Default Configuration

- **Bind Address**: localhost (127.0.0.1) only
- **External Access**: Blocked by default
- **Authentication**: PostgreSQL requires password (`adrian_password`)
- **Redis**: No authentication by default (localhost only)

### Hardening (Optional)

**Add Redis password:**
1. Edit: `C:\Program Files\Redis\redis.windows.conf`
2. Add: `requirepass your_secure_password`
3. Restart Redis service
4. Update `.env` file: `REDIS_PASSWORD=your_secure_password`

**Change PostgreSQL password:**
```powershell
# Using psql
& "C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres -c "ALTER USER adrian WITH PASSWORD 'new_password';"
```

Then update `.env` file.

---

## Performance Tips

### Redis Optimization

**Increase max memory (if needed):**
Edit `redis.windows.conf`:
```
maxmemory 256mb
maxmemory-policy allkeys-lru
```

### PostgreSQL Optimization

**Tune for development (optional):**
Edit `C:\Program Files\PostgreSQL\15\data\postgresql.conf`:
```
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
```

Restart service after changes:
```powershell
Restart-Service postgresql-x64-15
```

---

## FAQ

**Q: Will these services slow down my computer?**  
A: No. Combined they use ~60MB RAM when idle. Very lightweight.

**Q: Do I need to start them manually after reboot?**  
A: No. They're configured as Automatic startup type and will start with Windows.

**Q: Can I use different ports?**  
A: Yes, but you'll need to reconfigure the services and update ADRIAN's `.env` file.

**Q: Are they secure?**  
A: Yes. By default, they only listen on localhost (127.0.0.1), blocking external access.

**Q: Can I uninstall them later?**  
A: Yes, easily. Use Windows "Add or Remove Programs" or Chocolatey uninstall.

**Q: What if I don't want them to auto-start?**  
A: Set startup type to Manual: `Set-Service redis-server -StartupType Manual`

---

## Summary

✅ **Install once, forget about it**  
✅ **Auto-starts on boot**  
✅ **Lightweight and fast**  
✅ **Standard Windows services**  
✅ **No Docker complexity**

**This is the only supported infrastructure approach for ADRIAN.**

