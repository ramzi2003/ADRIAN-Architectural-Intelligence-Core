# A.D.R.I.A.N

**Autonomous Dreams Rest In Ambitions Naturally**

A sophisticated, human-like AI assistant designed to function as a personal digital butler, inspired by J.A.R.V.I.S. from Iron Man. ADRIAN integrates advanced voice recognition, natural language processing, and AI-driven decision-making to provide a seamless, intelligent interface for controlling computer systems.

---

## 🎯 Project Vision

ADRIAN is a local-first, microservice-based AI personal assistant that:
- Operates **offline-first** with optional cloud fallbacks
- Provides **human-like personality** and witty interactions
- Offers **deep system integration** for application and OS control
- Learns and adapts over time through **memory and context**
- Ensures **privacy and security** with local data storage

---

## 🏗️ Architecture

ADRIAN uses a **microservices architecture** where each service handles a specific responsibility:

| Service | Responsibility | Port |
|---------|---------------|------|
| **IO Service** | Voice/text input, STT | 8001 |
| **Processing Service** | NLP, intent classification, LLM | 8002 |
| **Memory Service** | PostgreSQL, FAISS, semantic search | 8003 |
| **Execution Service** | System control, integrations | 8004 |
| **Security Service** | Authentication, permissions | 8005 |
| **Output Service** | TTS, notifications | 8006 |

All services communicate via **Redis Pub/Sub** for asynchronous events and **REST/gRPC** for synchronous queries.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama (for local LLM)

### Installation

```powershell
# 1. Clone the repository
git clone <repository-url>
cd A.D.R.I.A.N

# 2. Install infrastructure (Redis, PostgreSQL) - Run as Administrator, ONE-TIME ONLY
.\scripts\install_infrastructure.ps1

# 3. Run setup (Python venv & dependencies)
.\scripts\setup.ps1

# 4. Install and run Ollama
ollama pull mistral:7b

# 5. Start all services
.\scripts\start_services.ps1

# 6. (Optional) Register auto-start - Run as Administrator
.\scripts\register_autostart.ps1

# 7. Test health
.\scripts\test_services.ps1
```

**Note**: Redis and PostgreSQL are installed as Windows services that auto-start on boot. No Docker required!

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## 📚 Documentation

- **[Setup Guide](SETUP_GUIDE.md)** - Complete installation and configuration guide
- **[Architecture Document](docs/architecture.md)** - Detailed system architecture
- **[Task 1: Architecture](docs/task1_architecture.md)** - Architecture decisions and diagrams

---

## 🧪 Testing the System

Send a text command to ADRIAN:

```powershell
$body = @{text = "open chrome"; user_id = "test_user"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8001/input/text -Method POST -Body $body -ContentType "application/json"
```

Watch the logs across service windows to see the event flow!

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python, FastAPI |
| **Message Bus** | Redis Pub/Sub |
| **Database** | PostgreSQL |
| **Vector Store** | FAISS |
| **Local LLM** | Mistral 7B (via Ollama) |
| **Cloud LLM** | Google Gemini |
| **Embeddings** | Sentence Transformers |

---

## 📁 Project Structure

```
A.D.R.I.A.N/
├── services/              # Microservices
│   ├── io_service/        # Input layer
│   ├── processing_service/# NLP & reasoning
│   ├── memory_service/    # Storage & retrieval
│   ├── execution_service/ # System control
│   ├── security_service/  # Auth & permissions
│   └── output_service/    # TTS & UI
├── shared/                # Shared utilities
│   ├── config.py          # Configuration management
│   ├── schemas.py         # Message schemas
│   ├── redis_client.py    # Redis wrapper
│   └── logging_config.py  # Logging setup
├── scripts/               # Orchestration scripts
│   ├── setup.ps1          # Initial setup
│   ├── start_services.ps1 # Start all services
│   ├── stop_services.ps1  # Stop all services
│   └── test_services.ps1  # Health checks
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── env.example            # Environment template
```

---

## 🎯 Current Status

**Phase 1 - Task 3: Code Scaffolding ✅ COMPLETE**

All 6 microservices are implemented as **working stubs** with:
- ✅ FastAPI applications with health endpoints
- ✅ Redis pub/sub integration
- ✅ Message schemas and communication protocols
- ✅ Structured logging to SQLite
- ✅ Configuration management
- ✅ Native Windows services for Redis & PostgreSQL
- ✅ PowerShell orchestration scripts

**Next Phase**: Implement voice input and STT (Phase 2)

---

## 🔮 Roadmap

- [x] **Task 1**: Architecture documentation
- [x] **Task 2**: Tech stack finalization  
- [x] **Task 3**: Code scaffolding (current)
- [ ] **Phase 2**: Input/Output Layer (Voice, STT, TTS)
- [ ] **Phase 3**: Processing Engine (NLP, LLM integration)
- [ ] **Phase 4**: Memory & Learning
- [ ] **Phase 5**: Security Framework
- [ ] **Phase 6**: Execution & Integrations
- [ ] **Phase 7**: Personality Engine
- [ ] **Phase 8**: Testing & Optimization

Target completion: **November 2026**

---

## 🤝 Contributing

This is a personal project currently in active development. Contributions, suggestions, and feedback are welcome as the project evolves.

---


## 🙏 Acknowledgments

Inspired by J.A.R.V.I.S. from the Iron Man franchise and the vision of creating a truly intelligent, adaptive, and personalized AI assistant.

---

**"Sir, I'm ready when you are."** - ADRIAN
