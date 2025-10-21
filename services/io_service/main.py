"""
IO Service - Input Layer
Handles voice and text input, hotword detection, and speech-to-text.
"""
import asyncio
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import UtteranceEvent, HealthResponse
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Import audio and hotword utilities
from .audio_utils import AudioStream
from .hotword_utils import test_hotword_detection

# Setup logging
logger = setup_logging("io-service")
settings = get_settings()

# Global audio stream for hotword detection
audio_stream: AudioStream = None


# Request models
class TextInputRequest(BaseModel):
    """Request model for text input."""
    text: str
    user_id: str = "default_user"


# Hotword detection callback
def on_hotword_detected():
    """Called when 'ADRIAN' hotword is detected."""
    try:
        logger.info("ADRIAN hotword detected! Starting command processing...")
        
        # Schedule async Redis publishing
        asyncio.create_task(_publish_hotword_event())
        
    except Exception as e:
        logger.error(f"Error in hotword callback: {e}")
        # This will trigger the smart error handling we implemented


async def _publish_hotword_event():
    """Async helper to publish hotword event to Redis."""
    try:
        # Get Redis client to publish activation event
        redis_client = await get_redis_client()
        
        # Create activation event
        activation_event = {
            "event_type": "hotword_detected",
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "default_user"
        }
        
        # Publish to Redis for processing service to handle
        await redis_client.publish("hotword_events", activation_event)
        
        logger.info("Hotword detection event published")
        
    except Exception as e:
        logger.error(f"Error publishing hotword event: {e}")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting IO Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    
    # Initialize audio stream with hotword detection
    global audio_stream
    try:
        logger.info("Initializing audio stream with hotword detection...")
        audio_stream = AudioStream(
            callback=lambda _: None,  # Dummy callback for now
            enable_vad=True,
            enable_hotword=True,
            hotword_sensitivity=0.7,
            hotword_callback=on_hotword_detected
        )
        logger.info("Audio stream initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize audio stream: {e}")
        audio_stream = None
    
    # TODO: Initialize STT engine (Whisper/Vosk)
    
    yield
    
    # Shutdown
    logger.info("Shutting down IO Service...")
    
    # Stop audio stream
    if audio_stream:
        audio_stream.stop()
        logger.info("Audio stream stopped")
    
    await redis_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="ADRIAN IO Service",
    description="Input Layer - Voice and Text Input Processing",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check hotword detection status
    hotword_status = "disabled"
    if audio_stream and audio_stream.is_hotword_enabled():
        status = audio_stream.get_hotword_status()
        hotword_status = "enabled" if status.get("is_running", False) else "stopped"
    
    return HealthResponse(
        service_name="io-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "stt_engine": "stub",
            "microphone": "available" if audio_stream else "unavailable",
            "hotword_detection": hotword_status
        }
    )


@app.post("/input/text")
async def text_input(request: TextInputRequest):
    """
    Accept text input and emit utterance event.
    This is a fallback for when voice input is not available.
    """
    try:
        redis_client = await get_redis_client()
        
        # Create utterance event
        utterance = UtteranceEvent(
            message_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            text=request.text,
            confidence=1.0,
            source="text",
            user_id=request.user_id
        )
        
        # Publish to Redis
        await redis_client.publish("utterances", utterance)
        
        logger.info(f"Text input received: '{request.text}'")
        
        return {
            "status": "success",
            "message": "Utterance published",
            "correlation_id": utterance.correlation_id
        }
    except Exception as e:
        logger.error(f"Error processing text input: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/input/voice/start")
async def start_voice_input():
    """
    Start voice input listening with hotword detection.
    """
    global audio_stream
    
    try:
        if audio_stream is None:
            # Initialize if not already done
            audio_stream = AudioStream(
                callback=lambda _: None,
                enable_vad=True,
                enable_hotword=True,
                hotword_sensitivity=0.7,
                hotword_callback=on_hotword_detected
            )
        
        if audio_stream.is_running:
            return {
                "status": "already_running",
                "message": "Voice input already active"
            }
        
        # Start the audio stream
        audio_stream.start()
        
        # Get status
        hotword_status = audio_stream.get_hotword_status()
        
        logger.info("Voice input started with hotword detection")
        return {
            "status": "started",
            "message": "Voice input and hotword detection active",
            "hotword_status": hotword_status
        }
        
    except Exception as e:
        logger.error(f"Error starting voice input: {e}")
        return {
            "status": "error",
            "message": f"Failed to start voice input: {str(e)}"
        }


@app.post("/input/voice/stop")
async def stop_voice_input():
    """
    Stop voice input listening.
    """
    global audio_stream
    
    try:
        if audio_stream is None or not audio_stream.is_running:
            return {
                "status": "already_stopped",
                "message": "Voice input not currently running"
            }
        
        # Stop the audio stream
        audio_stream.stop()
        
        logger.info("Voice input stopped")
        return {
            "status": "stopped",
            "message": "Voice input and hotword detection stopped"
        }
        
    except Exception as e:
        logger.error(f"Error stopping voice input: {e}")
        return {
            "status": "error",
            "message": f"Failed to stop voice input: {str(e)}"
        }


@app.get("/status/hotword")
async def get_hotword_status():
    """
    Get current hotword detection status.
    """
    global audio_stream
    
    try:
        if audio_stream is None:
            return {
                "status": "not_initialized",
                "message": "Audio stream not initialized",
                "hotword_enabled": False
            }
        
        hotword_status = audio_stream.get_hotword_status()
        
        return {
            "status": "ok",
            "hotword_enabled": audio_stream.is_hotword_enabled(),
            "audio_stream_running": audio_stream.is_running,
            "hotword_details": hotword_status
        }
        
    except Exception as e:
        logger.error(f"Error getting hotword status: {e}")
        return {
            "status": "error",
            "message": f"Error getting status: {str(e)}"
        }


# Stub functions for future implementation
async def initialize_microphone():
    """TODO: Initialize microphone and audio capture."""
    logger.info("[STUB] Initializing microphone...")
    pass


async def initialize_hotword_detection():
    """TODO: Initialize hotword detection (Porcupine/custom)."""
    logger.info("[STUB] Initializing hotword detection...")
    pass


async def initialize_stt_engine():
    """TODO: Initialize Speech-to-Text engine (Whisper/Vosk)."""
    logger.info("[STUB] Initializing STT engine...")
    pass


async def process_audio_stream():
    """
    TODO: Main loop for processing audio stream.
    - Capture audio from microphone
    - Detect hotword ("ADRIAN")
    - When detected, capture utterance
    - Send to STT
    - Emit UtteranceEvent
    """
    logger.info("[STUB] Audio processing loop not yet implemented")
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.io_service_port,
        log_level=settings.log_level.lower()
    )

