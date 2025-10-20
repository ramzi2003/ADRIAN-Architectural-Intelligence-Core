"""
Memory Service - Storage and Retrieval Layer
Handles short-term memory, long-term storage (PostgreSQL), and semantic search (FAISS).
"""
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from shared.config import get_settings
from shared.schemas import HealthResponse, MemoryQuery, MemoryResult
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Import database and FAISS operations
from services.memory_service.database import (
    get_pool, close_pool, get_user_by_username, create_user,
    store_conversation, get_recent_conversations,
    store_memory_embedding, get_memories_by_indices,
    get_preference, set_preference, get_all_preferences
)
from services.memory_service.faiss_ops import (
    load_index, save_index, add_to_index, search_index, 
    generate_embedding, get_index_stats
)

# Setup logging
logger = setup_logging("memory-service")
settings = get_settings()


# Request models
class StoreMemoryRequest(BaseModel):
    """Request to store a memory entry."""
    text: str
    metadata: Dict[str, Any] = {}
    user_id: str = "default_user"


class SearchRequest(BaseModel):
    """Request to search memories."""
    query: str
    limit: int = 5
    user_id: str = "default_user"


# In-memory short-term storage (session context)
short_term_memory: Dict[str, List[Dict]] = {}


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Memory Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    
    # Initialize PostgreSQL connection pool
    await get_pool()
    logger.info("PostgreSQL connection pool initialized")
    
    # Initialize or load FAISS index
    load_index()
    logger.info("FAISS index loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Memory Service...")
    
    # Save FAISS index
    save_index()
    logger.info("FAISS index saved")
    
    # Close PostgreSQL connection
    await close_pool()
    logger.info("PostgreSQL connection pool closed")
    
    await redis_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="ADRIAN Memory Service",
    description="Storage and Retrieval Layer with Semantic Search",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check PostgreSQL
    try:
        pool = await get_pool()
        pg_status = "connected" if pool else "disconnected"
    except Exception:
        pg_status = "error"
    
    # Check FAISS
    faiss_stats = get_index_stats()
    faiss_status = "loaded" if faiss_stats["loaded"] else "not_loaded"
    
    return HealthResponse(
        service_name="memory-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "postgresql": pg_status,
            "faiss": faiss_status,
            "faiss_vectors": str(faiss_stats.get("total_vectors", 0))
        }
    )


@app.post("/memory/store")
async def store_memory(request: StoreMemoryRequest):
    """
    Store a memory entry in both PostgreSQL and FAISS index.
    """
    try:
        # Get or create user
        user = await get_user_by_username(request.user_id)
        if not user:
            from uuid import UUID
            user_uuid = await create_user(request.user_id)
        else:
            user_uuid = user['id']
        
        # Add to FAISS index and get the index ID
        embedding_index = await add_to_index(request.text)
        
        # Store metadata in PostgreSQL
        memory_id = await store_memory_embedding(
            user_id=user_uuid,
            text=request.text,
            embedding_index=embedding_index,
            source_type=request.metadata.get("source_type", "manual"),
            metadata=request.metadata
        )
        
        logger.info(f"Stored memory: {memory_id} at FAISS index {embedding_index}")
        
        return {
            "status": "success",
            "memory_id": str(memory_id),
            "embedding_index": embedding_index
        }
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search")
async def search_memory(request: SearchRequest):
    """
    Semantic search through stored memories using FAISS.
    """
    try:
        # Search FAISS index for similar vectors
        results = await search_index(request.query, k=request.limit)
        
        if not results:
            logger.info(f"No results found for query: '{request.query}'")
            return {
                "status": "success",
                "query": request.query,
                "results": []
            }
        
        # Extract embedding indices and distances
        indices = [r[0] for r in results]
        distances = [r[1] for r in results]
        
        # Fetch full entries from PostgreSQL
        memories = await get_memories_by_indices(indices)
        
        # Combine with distance scores
        for i, memory in enumerate(memories):
            memory['similarity_score'] = float(1.0 / (1.0 + distances[i]))  # Convert distance to similarity
            memory['distance'] = float(distances[i])
        
        # Sort by similarity (highest first)
        memories.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Search for '{request.query}' returned {len(memories)} results")
        
        return {
            "status": "success",
            "query": request.query,
            "results": memories
        }
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/session/{user_id}")
async def get_session_memory(user_id: str):
    """
    Get short-term session memory for a user.
    """
    return {
        "user_id": user_id,
        "session_memory": short_term_memory.get(user_id, [])
    }


@app.post("/memory/session/{user_id}")
async def add_to_session(user_id: str, entry: Dict[str, Any]):
    """
    Add an entry to short-term session memory.
    """
    if user_id not in short_term_memory:
        short_term_memory[user_id] = []
    
    short_term_memory[user_id].append({
        "timestamp": datetime.utcnow().isoformat(),
        "entry": entry
    })
    
    # Keep only last 50 entries per user
    short_term_memory[user_id] = short_term_memory[user_id][-50:]
    
    return {"status": "success", "session_size": len(short_term_memory[user_id])}


@app.delete("/memory/session/{user_id}")
async def clear_session(user_id: str):
    """
    Clear short-term session memory for a user.
    """
    if user_id in short_term_memory:
        del short_term_memory[user_id]
    
    return {"status": "success", "message": "Session cleared"}


@app.post("/conversation/store")
async def store_conversation_message(
    user_id: str,
    message: str,
    message_type: str,
    intent: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Store a conversation message in the database."""
    try:
        # Get or create user
        user = await get_user_by_username(user_id)
        if not user:
            user_uuid = await create_user(user_id)
        else:
            user_uuid = user['id']
        
        # Store conversation
        conversation_id = await store_conversation(
            user_id=user_uuid,
            message=message,
            message_type=message_type,
            intent=intent,
            correlation_id=correlation_id,
            metadata=metadata
        )
        
        return {
            "status": "success",
            "conversation_id": str(conversation_id)
        }
    except Exception as e:
        logger.error(f"Error storing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/recent/{user_id}")
async def get_recent_conversation_history(user_id: str, limit: int = 10):
    """Get recent conversation history for a user."""
    try:
        user = await get_user_by_username(user_id)
        if not user:
            return {"status": "success", "conversations": []}
        
        conversations = await get_recent_conversations(user['id'], limit)
        
        # Convert UUIDs to strings for JSON serialization
        for conv in conversations:
            conv['id'] = str(conv['id'])
            conv['user_id'] = str(conv['user_id'])
            if conv.get('session_id'):
                conv['session_id'] = str(conv['session_id'])
        
        return {
            "status": "success",
            "conversations": conversations
        }
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_memory_stats():
    """Get memory service statistics."""
    try:
        faiss_stats = get_index_stats()
        
        # Get database stats
        pool = await get_pool()
        async with pool.acquire() as conn:
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
            conversation_count = await conn.fetchval("SELECT COUNT(*) FROM conversations")
            memory_count = await conn.fetchval("SELECT COUNT(*) FROM memory_embeddings")
        
        return {
            "status": "success",
            "stats": {
                "users": user_count,
                "conversations": conversation_count,
                "memories": memory_count,
                "faiss_vectors": faiss_stats.get("total_vectors", 0),
                "faiss_loaded": faiss_stats.get("loaded", False)
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.memory_service_port,
        log_level=settings.log_level.lower()
    )

