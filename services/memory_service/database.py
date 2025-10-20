"""
PostgreSQL database operations for memory service.
Handles connections and CRUD operations for all tables.
"""
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID, uuid4
import json

from shared.config import get_settings, get_postgres_url
from shared.logging_config import get_logger

logger = get_logger("memory-service.database")
settings = get_settings()

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create connection pool."""
    global _pool
    if _pool is None:
        # asyncpg doesn't use the +asyncpg suffix in the URL
        connection_url = get_postgres_url()  # Use regular postgres URL
        _pool = await asyncpg.create_pool(
            connection_url,
            min_size=2,
            max_size=10
        )
        logger.info("PostgreSQL connection pool created")
    return _pool


async def close_pool():
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


# ============================================================================
# User Operations
# ============================================================================

async def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1",
            username
        )
        return dict(row) if row else None


async def get_user_by_id(user_id: UUID) -> Optional[Dict]:
    """Get user by ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE id = $1",
            user_id
        )
        return dict(row) if row else None


async def create_user(username: str, email: Optional[str] = None, metadata: Optional[Dict] = None) -> UUID:
    """Create a new user."""
    pool = await get_pool()
    user_id = uuid4()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, username, email, metadata)
            VALUES ($1, $2, $3, $4)
            """,
            user_id, username, email, json.dumps(metadata) if metadata else None
        )
    
    logger.info(f"Created user: {username} ({user_id})")
    return user_id


# ============================================================================
# Session Operations
# ============================================================================

async def create_session(user_id: UUID, metadata: Optional[Dict] = None) -> UUID:
    """Create a new session."""
    pool = await get_pool()
    session_id = uuid4()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (id, user_id, session_metadata)
            VALUES ($1, $2, $3)
            """,
            session_id, user_id, json.dumps(metadata) if metadata else None
        )
    
    logger.info(f"Created session: {session_id} for user {user_id}")
    return session_id


async def end_session(session_id: UUID):
    """End a session."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions 
            SET end_time = CURRENT_TIMESTAMP, is_active = FALSE
            WHERE id = $1
            """,
            session_id
        )
    
    logger.info(f"Ended session: {session_id}")


# ============================================================================
# Conversation Operations
# ============================================================================

async def store_conversation(
    user_id: UUID,
    message: str,
    message_type: str,
    intent: Optional[str] = None,
    confidence: Optional[float] = None,
    correlation_id: Optional[str] = None,
    session_id: Optional[UUID] = None,
    metadata: Optional[Dict] = None
) -> UUID:
    """Store a conversation message."""
    pool = await get_pool()
    conversation_id = uuid4()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO conversations 
            (id, user_id, session_id, message, message_type, intent, confidence, correlation_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            conversation_id, user_id, session_id, message, message_type, 
            intent, confidence, correlation_id, json.dumps(metadata) if metadata else None
        )
    
    logger.info(f"Stored conversation: {conversation_id}")
    return conversation_id


async def get_recent_conversations(user_id: UUID, limit: int = 10) -> List[Dict]:
    """Get recent conversations for a user."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM conversations
            WHERE user_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            user_id, limit
        )
    
    return [dict(row) for row in rows]


# ============================================================================
# Memory Embedding Operations
# ============================================================================

async def store_memory_embedding(
    user_id: UUID,
    text: str,
    embedding_index: int,
    source_type: str,
    source_id: Optional[UUID] = None,
    metadata: Optional[Dict] = None
) -> UUID:
    """Store memory embedding metadata."""
    pool = await get_pool()
    memory_id = uuid4()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_embeddings 
            (id, user_id, text, embedding_index, source_type, source_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            memory_id, user_id, text, embedding_index, source_type, source_id,
            json.dumps(metadata) if metadata else None
        )
    
    logger.info(f"Stored memory embedding: {memory_id} at index {embedding_index}")
    return memory_id


async def get_memory_by_embedding_index(embedding_index: int) -> Optional[Dict]:
    """Get memory metadata by FAISS index."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM memory_embeddings WHERE embedding_index = $1",
            embedding_index
        )
    
    return dict(row) if row else None


async def get_memories_by_indices(embedding_indices: List[int]) -> List[Dict]:
    """Get multiple memories by FAISS indices."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM memory_embeddings WHERE embedding_index = ANY($1)",
            embedding_indices
        )
    
    return [dict(row) for row in rows]


# ============================================================================
# Preference Operations
# ============================================================================

async def get_preference(user_id: UUID, category: str, key: str) -> Optional[Any]:
    """Get a user preference."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT value FROM preferences WHERE user_id = $1 AND category = $2 AND key = $3",
            user_id, category, key
        )
    
    return row['value'] if row else None


async def set_preference(user_id: UUID, category: str, key: str, value: Any):
    """Set a user preference."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO preferences (user_id, category, key, value)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, category, key)
            DO UPDATE SET value = $4, updated_at = CURRENT_TIMESTAMP
            """,
            user_id, category, key, json.dumps(value)
        )
    
    logger.info(f"Set preference: {category}.{key} for user {user_id}")


async def get_all_preferences(user_id: UUID) -> Dict[str, Dict]:
    """Get all preferences for a user, grouped by category."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT category, key, value FROM preferences WHERE user_id = $1",
            user_id
        )
    
    result = {}
    for row in rows:
        category = row['category']
        if category not in result:
            result[category] = {}
        result[category][row['key']] = row['value']
    
    return result


# ============================================================================
# Task Operations
# ============================================================================

async def create_task(
    user_id: UUID,
    title: str,
    description: Optional[str] = None,
    task_type: str = "reminder",
    scheduled_time: Optional[datetime] = None,
    metadata: Optional[Dict] = None
) -> UUID:
    """Create a new task."""
    pool = await get_pool()
    task_id = uuid4()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO tasks 
            (id, user_id, title, description, task_type, scheduled_time, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            task_id, user_id, title, description, task_type, scheduled_time,
            json.dumps(metadata) if metadata else None
        )
    
    logger.info(f"Created task: {task_id}")
    return task_id


async def get_pending_tasks(user_id: UUID) -> List[Dict]:
    """Get all pending tasks for a user."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM tasks
            WHERE user_id = $1 AND is_completed = FALSE
            ORDER BY scheduled_time ASC NULLS LAST
            """,
            user_id
        )
    
    return [dict(row) for row in rows]


# ============================================================================
# System Logs
# ============================================================================

async def log_system_event(
    event_type: str,
    event_data: Dict,
    user_id: Optional[UUID] = None,
    severity: str = "info"
):
    """Log a system event."""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO system_logs (user_id, event_type, event_data, severity)
            VALUES ($1, $2, $3, $4)
            """,
            user_id, event_type, json.dumps(event_data), severity
        )

