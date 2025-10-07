"""
Output Service - TTS and UI Layer
Handles text-to-speech, visual output, and system notifications.
"""
import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import ResponseText, ActionResult, HealthResponse
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Setup logging
logger = setup_logging("output-service")
settings = get_settings()


# Request models
class SpeakRequest(BaseModel):
    """Request to speak text."""
    text: str
    voice: str = "default"


# Background task flags
_listening_responses = False
_listening_results = False


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _listening_responses, _listening_results
    logger.info("Starting Output Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    
    # Start background listeners
    _listening_responses = True
    _listening_results = True
    asyncio.create_task(listen_for_responses())
    asyncio.create_task(listen_for_action_results())
    
    # TODO: Initialize TTS engine
    await initialize_tts()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Output Service...")
    _listening_responses = False
    _listening_results = False
    await redis_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="ADRIAN Output Service",
    description="TTS and UI Layer",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        service_name="output-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "tts_engine": "stub"
        }
    )


@app.post("/speak")
async def speak(request: SpeakRequest):
    """
    Synchronous endpoint to speak text directly.
    Useful for testing without going through Redis.
    """
    try:
        await text_to_speech(request.text)
        return {"status": "success", "text": request.text}
    except Exception as e:
        logger.error(f"Error speaking text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def listen_for_responses():
    """
    Background task that listens to response_text events from Redis.
    """
    logger.info("Started listening for responses...")
    redis_client = await get_redis_client()
    
    async def handle_response(data: dict):
        """Process incoming response text."""
        try:
            response = ResponseText(**data)
            logger.info(f"Received response: '{response.text}'")
            
            # Display text (console for now)
            print(f"\n[ADRIAN]: {response.text}\n")
            
            # Speak if requested
            if response.should_speak:
                await text_to_speech(response.text)
            
        except Exception as e:
            logger.error(f"Error handling response: {e}")
    
    # Subscribe and listen
    await redis_client.subscribe("responses", handle_response)


async def listen_for_action_results():
    """
    Background task that listens to action_result events from Redis.
    """
    logger.info("Started listening for action results...")
    redis_client = await get_redis_client()
    
    async def handle_action_result(data: dict):
        """Process incoming action result."""
        try:
            result = ActionResult(**data)
            
            if result.success:
                logger.info(f"Action completed successfully: {result.action_id}")
                # Optionally provide feedback
            else:
                logger.error(f"Action failed: {result.error_message}")
                # Could speak error message
                
        except Exception as e:
            logger.error(f"Error handling action result: {e}")
    
    # Subscribe and listen
    await redis_client.subscribe("action_results", handle_action_result)


async def initialize_tts():
    """
    TODO: Initialize text-to-speech engine.
    Options: pyttsx3, Coqui TTS, ElevenLabs API, etc.
    """
    logger.info("[STUB] Initializing TTS engine...")
    pass


async def text_to_speech(text: str):
    """
    Convert text to speech and play audio.
    TODO: Implement actual TTS.
    """
    logger.info(f"[STUB] Speaking: '{text}'")
    
    # Stub: just log for now
    # In real implementation:
    # 1. Generate audio using TTS engine
    # 2. Play audio through speakers
    # 3. Handle playback queue for multiple messages
    
    pass


async def show_notification(title: str, message: str):
    """
    TODO: Show system notification.
    Platform-specific implementation needed.
    """
    logger.info(f"[STUB] Notification: {title} - {message}")
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.output_service_port,
        log_level=settings.log_level.lower()
    )

