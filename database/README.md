# ADRIAN Database Schema

This directory contains database initialization scripts for ADRIAN's memory system.

## Files

- **`init_database.sql`** - PostgreSQL schema creation script
- **`init_faiss_index.py`** - FAISS vector index initialization script

## Database Tables

### Core Tables

1. **`users`** - User profiles and authentication
   - Stores username, email, voice biometrics
   - Primary key: UUID

2. **`sessions`** - User interaction sessions
   - Tracks session start/end times
   - Links to user, stores session context

3. **`conversations`** - Complete chat history
   - Stores all messages (user, assistant, system)
   - Includes intent classification and correlation IDs

4. **`memory_embeddings`** - Vector embedding metadata
   - Links text to FAISS vector indices
   - Enables semantic search
   - Tracks source (conversation, note, document)

5. **`preferences`** - User preferences and settings
   - Key-value store for learned behaviors
   - Organized by category

6. **`tasks`** - Scheduled tasks and reminders
   - Supports recurring tasks
   - Tracks completion status

7. **`system_logs`** - Audit trail
   - System events and actions
   - Severity levels for filtering

### Views

- **`v_recent_conversations`** - Recent messages with user info
- **`v_active_sessions`** - Active sessions with message counts

## Initialization

### Run Once (After PostgreSQL is installed):

```powershell
.\scripts\init_database.ps1
```

This script will:
1. Create all tables in the `adrian` database
2. Create indexes for performance
3. Create views
4. Insert default user
5. Initialize FAISS vector index
6. Verify everything was created successfully

## Manual Commands

### Create Tables Only

```powershell
$pgPath = "C:\Program Files\PostgreSQL\17\bin"
& "$pgPath\psql.exe" -U adrian -d adrian -f database\init_database.sql
```

### Initialize FAISS Only

```powershell
$env:PYTHONPATH = $PWD
.\venv\Scripts\python.exe database\init_faiss_index.py
```

### Verify Tables

```powershell
& "$pgPath\psql.exe" -U adrian -d adrian -c "\dt"
```

### Query Tables

```sql
-- List all users
SELECT * FROM users;

-- Recent conversations
SELECT * FROM v_recent_conversations LIMIT 10;

-- Memory stats
SELECT COUNT(*) as total_memories FROM memory_embeddings;

-- FAISS index stats
SELECT 
    COUNT(*) as total_embeddings,
    COUNT(DISTINCT user_id) as unique_users,
    MAX(created_at) as latest_memory
FROM memory_embeddings;
```

## FAISS Index

- **Location**: `data/faiss_index/index.faiss`
- **Metadata**: `data/faiss_index/metadata.json`
- **Dimension**: 384 (all-MiniLM-L6-v2 model)
- **Type**: IndexIDMap(IndexFlatL2) - Exact search with L2 distance

## Architecture

```
User Interaction
      ↓
Memory Service API
      ↓
   ┌──┴──┐
   ↓     ↓
PostgreSQL  FAISS
(metadata)  (vectors)
   ↓     ↓
Combined Results
   (semantic search)
```

When storing:
1. Text → Generate embedding (sentence-transformers)
2. Embedding → FAISS index (returns index ID)
3. Metadata + index ID → PostgreSQL

When searching:
1. Query → Generate embedding
2. Search FAISS for similar vectors (returns index IDs)
3. Fetch metadata from PostgreSQL using index IDs
4. Return combined results with similarity scores

## Schema Diagram

```sql
users
  ├── sessions (1:many)
  │     └── conversations (1:many)
  ├── memory_embeddings (1:many)
  ├── preferences (1:many)
  └── tasks (1:many)
```

## Notes

- All IDs use UUID v4 for security and distribution
- Timestamps use PostgreSQL CURRENT_TIMESTAMP
- JSONB used for flexible metadata storage
- Indexes created on frequently queried columns
- Automatic `updated_at` triggers on relevant tables

