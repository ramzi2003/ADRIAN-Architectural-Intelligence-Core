"""
Processing Service - NLP and Reasoning Layer
Handles intent classification, LLM interaction, and personality injection.
"""
import asyncio
import uuid
import httpx
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import (
    UtteranceEvent, ActionSpec, ResponseText, 
    HealthResponse, MemoryQuery
)
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Setup logging
logger = setup_logging("processing-service")
settings = get_settings()


# Request models
class ProcessRequest(BaseModel):
    """Request model for synchronous processing."""
    text: str
    user_id: str = "default_user"


# Background task flag
_listening = False


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _listening
    logger.info("Starting Processing Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    
    # Start background listener for utterances
    _listening = True
    asyncio.create_task(listen_for_utterances())
    
    # TODO: Initialize connection to Ollama
    # TODO: Initialize Gemini client (optional)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Processing Service...")
    _listening = False
    await redis_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="ADRIAN Processing Service",
    description="NLP and Reasoning Layer",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    ollama_status = await check_ollama_connection()
    
    return HealthResponse(
        service_name="processing-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "ollama": ollama_status,
            "gemini": "not_configured" if not settings.gemini_api_key else "ready"
        }
    )


@app.post("/process")
async def process_text(request: ProcessRequest):
    """
    Synchronous endpoint to process text and return intent/response.
    Useful for testing without going through Redis.
    """
    try:
        result = await process_utterance(request.text, request.user_id)
        return result
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def listen_for_utterances():
    """
    Background task that listens to utterance events from Redis.
    """
    logger.info("Started listening for utterances...")
    redis_client = await get_redis_client()
    
    async def handle_utterance(data: dict):
        """Process incoming utterance."""
        try:
            utterance = UtteranceEvent(**data)
            logger.info(f"Received utterance: '{utterance.text}' from {utterance.source}")
            
            # Process the utterance
            await process_utterance(
                utterance.text, 
                utterance.user_id or "default_user",
                utterance.correlation_id
            )
        except Exception as e:
            logger.error(f"Error handling utterance: {e}")
    
    # Subscribe and listen
    await redis_client.subscribe("utterances", handle_utterance)


async def process_utterance(text: str, user_id: str, correlation_id: Optional[str] = None):
    """
    Main processing pipeline:
    1. Query memory for context
    2. Classify intent
    3. Generate response with personality
    4. Emit action_spec and response_text
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    redis_client = await get_redis_client()
    
    # Step 1: Query memory service for context (STUB)
    # TODO: Make actual call to memory-service
    context = await query_memory_context(text, correlation_id)
    
    # Step 2: Classify intent (STUB)
    intent, parameters = await classify_intent(text, context)
    
    # Step 3: Generate response (STUB)
    response_text = await generate_response(text, intent, context)
    
    # Step 4: Apply personality/wit module (STUB)
    response_text = await apply_personality(response_text, intent)
    
    # Step 5: Emit action_spec if needed
    if intent != "conversation":
        action_spec = ActionSpec(
            message_id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            intent=intent,
            parameters=parameters,
            requires_permission=intent in ["delete_file", "shutdown", "install_app"],
            confidence=0.85
        )
        await redis_client.publish("action_specs", action_spec)
        logger.info(f"Published action_spec: {intent}")
    
    # Step 6: Emit response_text
    response = ResponseText(
        message_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        text=response_text,
        should_speak=True
    )
    await redis_client.publish("responses", response)
    logger.info(f"Published response: '{response_text}'")
    
    return {
        "intent": intent,
        "response": response_text,
        "correlation_id": correlation_id
    }


async def query_memory_context(text: str, correlation_id: str) -> dict:
    """
    Query memory service for relevant context.
    TODO: Implement actual HTTP call to memory-service.
    """
    logger.info(f"[STUB] Querying memory for context: '{text}'")
    return {"context": "stub_context", "history": []}


async def classify_intent(text: str, context: dict) -> tuple[str, dict]:
    """
    Classify user intent from text.
    TODO: Implement intent classification using local LLM or lightweight model.
    
    Possible intents:
    - open_app
    - close_app
    - search
    - reminder
    - file_operation
    - system_control
    - conversation
    """
    text_lower = text.lower()
    
    # Simple rule-based classification (STUB)
    if "open" in text_lower:
        app_name = text_lower.replace("open", "").strip()
        return "open_app", {"app_name": app_name}
    elif "close" in text_lower:
        app_name = text_lower.replace("close", "").strip()
        return "close_app", {"app_name": app_name}
    elif "search" in text_lower or "find" in text_lower:
        return "search", {"query": text}
    elif "remind" in text_lower:
        return "reminder", {"text": text}
    else:
        return "conversation", {}


async def generate_response(text: str, intent: str, context: dict) -> str:
    """
    Generate response text using LLM (Ollama or Gemini).
    TODO: Implement actual LLM call.
    """
    logger.info(f"[STUB] Generating response for intent: {intent}")
    
    # Try Ollama first (local)
    try:
        response = await call_ollama(text, context)
        return response
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}, falling back to stub")
        return f"Understood. Processing your request about: {text}"


async def apply_personality(response: str, intent: str) -> str:
    """
    Apply personality/wit module to make response more natural and Jarvis-like.
    TODO: Implement personality layer.
    """
    logger.info("[STUB] Applying personality layer")
    
    # Simple prefix for now
    if intent == "conversation":
        return f"Sir, {response}"
    else:
        return f"At once, Sir. {response}"


async def call_ollama(prompt: str, context: dict) -> str:
    """
    Call Ollama API for LLM inference.
    TODO: Implement full Ollama integration with proper prompting.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                raise Exception(f"Ollama returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        raise


async def check_ollama_connection() -> str:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            if response.status_code == 200:
                return "connected"
            else:
                return "unreachable"
    except Exception:
        return "unreachable"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.processing_service_port,
        log_level=settings.log_level.lower()
    )

