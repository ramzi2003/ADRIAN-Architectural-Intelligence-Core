# A.D.R.I.A.N Setup Guide

## Prerequisites

### Required Software

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure Python is added to PATH
   - Verify: `python --version`

2. **Ollama** (for local LLM)
   - Download from [ollama.ai](https://ollama.ai/)
   - After installation, pull the Mistral model:
     ```bash
     ollama pull mistral:7b
     ```

3. **Git**
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify: `git --version`

### Infrastructure (Installed Automatically)

The setup script will install these as **Windows services** (auto-start on boot):
- **Redis** - Message bus and caching
- **PostgreSQL** - Database for long-term memory

**Note**: No Docker required! Everything runs natively on Windows.

### Optional

- **Google Gemini API Key** (for cloud LLM fallback)
  - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## Installation Steps

### 1. Clone the Repository

```powershell
git clone <repository-url>
cd A.D.R.I.A.N
```

### 2. Install Infrastructure (One-Time, Requires Admin)

**Run PowerShell as Administrator**, then:

```powershell
.\scripts\install_infrastructure.ps1
```

This script will:
- Install Chocolatey (Windows package manager)
- Install Redis as a Windows service (auto-start)
- Install PostgreSQL as a Windows service (auto-start)
- Create the `adrian` database
- Configure everything to start automatically on boot

**Important**: This only needs to be run once!

### 3. Run Setup Script (Regular User)

```powershell
.\scripts\setup.ps1
```

This script will:
- Create a Python virtual environment
- Install all Python dependencies
- Create necessary directories (`logs`, `data`)
- Create `.env` file from template

### 4. Configure Environment

Edit the `.env` file and update if needed:

```env
# Optional: Add Gemini API key for cloud fallback
GEMINI_API_KEY=your_gemini_api_key_here

# Default PostgreSQL password (from installation)
POSTGRES_PASSWORD=adrian_password
```

### 5. Verify Infrastructure

```powershell
.\scripts\start_infrastructure.ps1
```

This checks if Redis and PostgreSQL services are running (starts them if stopped).

### 6. Start ADRIAN Services

```powershell
.\scripts\start_services.ps1
```

This opens 6 PowerShell windows, one for each microservice.

### 7. (Optional) Register Auto-Start

To make ADRIAN start automatically when you log in:

```powershell
# Run as Administrator
.\scripts\register_autostart.ps1
```

This creates a Task Scheduler task that starts all services automatically on login.

To remove auto-start:
```powershell
# Run as Administrator
.\scripts\unregister_autostart.ps1
```

### 8. Verify Services

```powershell
.\scripts\test_services.ps1
```

All services should report as "healthy".

---

## Quick Start Commands

| Action | Command | Requires Admin? |
|--------|---------|----------------|
| Install infrastructure | `.\scripts\install_infrastructure.ps1` | ✅ Yes (one-time) |
| Setup environment | `.\scripts\setup.ps1` | No |
| Check infrastructure | `.\scripts\start_infrastructure.ps1` | No |
| Start all services | `.\scripts\start_services.ps1` | No |
| Stop all services | `.\scripts\stop_services.ps1` | No |
| Register auto-start | `.\scripts\register_autostart.ps1` | ✅ Yes |
| Remove auto-start | `.\scripts\unregister_autostart.ps1` | ✅ Yes |
| Test health | `.\scripts\test_services.ps1` | No |

---

## Service Endpoints

| Service | Port | Health Check |
|---------|------|--------------|
| IO Service | 8001 | http://localhost:8001/health |
| Processing Service | 8002 | http://localhost:8002/health |
| Memory Service | 8003 | http://localhost:8003/health |
| Execution Service | 8004 | http://localhost:8004/health |
| Security Service | 8005 | http://localhost:8005/health |
| Output Service | 8006 | http://localhost:8006/health |

---

## Testing the System

### Test Text Input

Send a text command to ADRIAN:

```powershell
$body = @{
    text = "open chrome"
    user_id = "test_user"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8001/input/text -Method POST -Body $body -ContentType "application/json"
```

You should see:
1. IO Service logs the input
2. Processing Service classifies the intent
3. Execution Service receives the action (stub)
4. Output Service prints the response

---

## Troubleshooting

### Services Won't Start

**Issue**: "Port already in use"
- **Solution**: Stop existing services: `.\scripts\stop_services.ps1`

**Issue**: "Redis connection failed"
- **Solution**: Ensure Docker is running and infrastructure is started:
  ```powershell
  docker ps  # Should show adrian-redis and adrian-postgres
  ```

**Issue**: "Python not found"
- **Solution**: Ensure Python is in PATH and restart terminal

### Ollama Not Connected

**Issue**: Processing service shows "Ollama: unreachable"
- **Solution**: 
  1. Install Ollama from [ollama.ai](https://ollama.ai/)
  2. Pull the model: `ollama pull mistral:7b`
  3. Verify Ollama is running: `ollama list`

### Infrastructure Issues

**Issue**: "Redis service not found" or "PostgreSQL service not found"
- **Solution**: 
  1. Run PowerShell as Administrator
  2. Execute: `.\scripts\install_infrastructure.ps1`

**Issue**: "Access Denied" when installing infrastructure
- **Solution**: Right-click PowerShell and select "Run as Administrator"

---

## Development Tips

### Activating Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Installing New Dependencies

```powershell
.\venv\Scripts\pip install <package>
# Then update requirements.txt
.\venv\Scripts\pip freeze > requirements.txt
```

### Viewing Logs

Each service window shows live logs. Additionally, all logs are stored in:
- **SQLite Database**: `logs/adrian.db`

Query logs:
```sql
SELECT * FROM logs ORDER BY created_at DESC LIMIT 100;
```

### Running Individual Services

```powershell
.\venv\Scripts\python services\io_service\main.py
```

---

## Next Steps

After setup is complete and services are running:

1. **Test the system** with the example commands above
2. **Review the architecture** in `docs/architecture.md`
3. **Explore service APIs** by visiting `http://localhost:800X/docs` (FastAPI auto-docs)
4. **Begin Phase 2 implementation** (Voice input, STT)

---

## Architecture Overview

```
┌─────────────┐
│  User Input │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌──────────────┐
│   IO Service    │────▶│ Redis Pub/Sub│
└─────────────────┘     └───────┬──────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌───────────┐ ┌──────────┐ ┌─────────┐
            │Processing │ │Execution │ │ Output  │
            │  Service  │ │ Service  │ │ Service │
            └─────┬─────┘ └──────────┘ └─────────┘
                  │
            ┌─────┴──────┬────────────┐
            ▼            ▼            ▼
      ┌─────────┐  ┌─────────┐  ┌──────────┐
      │ Memory  │  │Security │  │PostgreSQL│
      │ Service │  │ Service │  │  + FAISS │
      └─────────┘  └─────────┘  └──────────┘
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review service logs in the individual PowerShell windows
3. Check the SQLite log database for detailed traces

---

**Status**: Task 3 Complete - Code Scaffolding ✅

All services are stub implementations ready for Phase 2 development.

