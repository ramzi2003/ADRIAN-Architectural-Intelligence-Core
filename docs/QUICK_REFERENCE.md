# ADRIAN Quick Reference

## Service URLs

| Service | Port | Health | Docs |
|---------|------|--------|------|
| IO | 8001 | http://localhost:8001/health | http://localhost:8001/docs |
| Processing | 8002 | http://localhost:8002/health | http://localhost:8002/docs |
| Memory | 8003 | http://localhost:8003/health | http://localhost:8003/docs |
| Execution | 8004 | http://localhost:8004/health | http://localhost:8004/docs |
| Security | 8005 | http://localhost:8005/health | http://localhost:8005/docs |
| Output | 8006 | http://localhost:8006/health | http://localhost:8006/docs |

## Quick Commands

```powershell
# Install infrastructure (ONE-TIME, as Administrator)
.\scripts\install_infrastructure.ps1

# Setup (first time only)
.\scripts\setup.ps1

# Check infrastructure (verifies Redis & PostgreSQL are running)
.\scripts\start_infrastructure.ps1

# Start all ADRIAN services
.\scripts\start_services.ps1

# Test health
.\scripts\test_services.ps1

# Stop all services
.\scripts\stop_services.ps1

# Register auto-start (as Administrator)
.\scripts\register_autostart.ps1
```

## Test Commands

### Send Text Input
```powershell
$body = @{text = "open chrome"; user_id = "test"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8001/input/text -Method POST -Body $body -ContentType "application/json"
```

### Test Processing Direct
```powershell
$body = @{text = "hello adrian"; user_id = "test"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8002/process -Method POST -Body $body -ContentType "application/json"
```

### Check Memory Session
```powershell
Invoke-RestMethod -Uri http://localhost:8003/memory/session/test_user
```

### Execute Action Direct
```powershell
$body = @{intent = "open_app"; parameters = @{app_name = "chrome"}} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8004/execute -Method POST -Body $body -ContentType "application/json"
```

## Environment Variables

Copy `env.example` to `.env` and configure:

```env
# Must Configure
POSTGRES_PASSWORD=your_secure_password

# Optional
GEMINI_API_KEY=your_api_key
OLLAMA_MODEL=mistral:7b
LOG_LEVEL=INFO
```

## Architecture

```
User Input
    ↓
IO Service (8001) → Redis Pub/Sub
    ↓
Processing Service (8002) ↔ Memory Service (8003)
    ↓
Security Service (8005)
    ↓
Execution Service (8004)
    ↓
Output Service (8006)
    ↓
User Output (TTS/Text)
```

## Redis Channels

| Channel | Purpose |
|---------|---------|
| `utterances` | User input events |
| `action_specs` | Action commands |
| `action_results` | Action outcomes |
| `responses` | Text responses to user |
| `security_checks` | Permission requests |
| `security_results` | Permission decisions |

## Log Database

Location: `logs/adrian.db`

```sql
-- View recent logs
SELECT * FROM logs ORDER BY created_at DESC LIMIT 50;

-- Trace a request
SELECT service_name, level, message, timestamp
FROM logs
WHERE correlation_id = 'your-correlation-id'
ORDER BY timestamp;

-- View errors
SELECT * FROM logs WHERE level = 'ERROR' ORDER BY created_at DESC;
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Run `.\scripts\stop_services.ps1` |
| Redis connection failed | Check service: `Get-Service redis-server` or run `.\scripts\install_infrastructure.ps1` |
| PostgreSQL connection failed | Check service: `Get-Service postgresql*` or run `.\scripts\install_infrastructure.ps1` |
| Ollama unreachable | Install Ollama and run `ollama pull mistral:7b` |
| Python not found | Add Python to PATH and restart terminal |
| Import errors | Run `.\venv\Scripts\pip install -r requirements.txt` |
| Access denied | Run PowerShell as Administrator for infrastructure/auto-start scripts |

## Next Steps

1. ✅ Task 3 Complete (Code Scaffolding)
2. → Phase 2: Implement Voice Input & TTS
3. → Phase 3: NLP & LLM Integration
4. → Phase 4: Memory & FAISS Implementation

---

**Quick Start**: 
1. `.\scripts\install_infrastructure.ps1` (as Admin, one-time)
2. `.\scripts\setup.ps1`
3. `.\scripts\start_services.ps1`

**Auto-Start**: `.\scripts\register_autostart.ps1` (as Admin) - Services start automatically on login!

