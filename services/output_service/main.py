"""
Output Service - TTS and UI Layer
Handles text-to-speech, visual output, and system notifications.
"""
import asyncio
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.config import get_settings
from shared.schemas import ResponseText, ActionResult, HealthResponse
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging
from shared.tts_processor import TTSProcessor, TTSConfig

# Setup logging
logger = setup_logging("output-service")
settings = get_settings()

# Global TTS processor
tts_processor: TTSProcessor = None


# Request models
class SpeakRequest(BaseModel):
    """Request to speak text."""
    text: str
    interrupt: bool = False  # Interrupt current speech if True


class VolumeRequest(BaseModel):
    """Request to set volume."""
    volume: float  # 0.0 to 1.0


class ToggleRequest(BaseModel):
    """Request to toggle TTS on/off."""
    enabled: bool


# Background task flags
_listening_responses = False
_listening_results = False


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _listening_responses, _listening_results, tts_processor
    logger.info("Starting Output Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    
    # Initialize TTS processor
    await initialize_tts()
    
    # Start background listeners
    _listening_responses = True
    _listening_results = True
    asyncio.create_task(listen_for_responses())
    asyncio.create_task(listen_for_action_results())
    
    print("\n" + "="*60)
    print("✅ Output Service ready - TTS enabled!")
    print("="*60 + "\n")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Output Service...")
    _listening_responses = False
    _listening_results = False
    
    # Shutdown TTS processor
    if tts_processor:
        tts_processor.shutdown()
    
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
    global tts_processor
    
    tts_status = "unavailable"
    if tts_processor:
        stats = tts_processor.get_performance_stats()
        if stats["model_loaded"]:
            tts_status = f"loaded ({stats['model_name']})"
        else:
            tts_status = "not_loaded"
    
    return HealthResponse(
        service_name="output-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "tts_engine": tts_status
        }
    )


@app.post("/speak")
async def speak(request: SpeakRequest):
    """
    Synchronous endpoint to speak text directly.
    Useful for testing without going through Redis.
    """
    global tts_processor
    
    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")
    
    try:
        success = tts_processor.speak(request.text, blocking=False, interrupt=request.interrupt)
        if success:
            return {"status": "success", "text": request.text, "queued": not request.interrupt}
        else:
            raise HTTPException(status_code=500, detail="Failed to speak text")
    except Exception as e:
        logger.error(f"Error speaking text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/volume")
async def set_volume(request: VolumeRequest):
    """
    Set TTS volume level.
    """
    global tts_processor
    
    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")
    
    if not (0.0 <= request.volume <= 1.0):
        raise HTTPException(status_code=400, detail="Volume must be between 0.0 and 1.0")
    
    tts_processor.set_volume(request.volume)
    return {"status": "success", "volume": request.volume}


@app.get("/tts/volume")
async def get_volume():
    """Get current TTS volume level."""
    global tts_processor
    
    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")
    
    stats = tts_processor.get_performance_stats()
    return {"volume": stats["volume"]}


@app.post("/tts/toggle")
async def toggle_tts(request: ToggleRequest):
    """
    Enable or disable TTS.
    """
    global tts_processor
    
    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")
    
    tts_processor.set_enabled(request.enabled)
    return {"status": "success", "enabled": request.enabled}


@app.get("/tts/status")
async def get_tts_status():
    """Get TTS status and statistics."""
    global tts_processor
    
    if not tts_processor:
        return {
            "available": False,
            "message": "TTS processor not initialized"
        }
    
    stats = tts_processor.get_performance_stats()
    return {
        "available": True,
        **stats
    }


@app.post("/tts/stop")
async def stop_tts():
    """Stop current TTS playback and clear queue (for interruption)."""
    global tts_processor
    
    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")
    
    try:
        tts_processor.stop_playback()
        logger.info("TTS playback stopped (interrupted)")
        return {"status": "success", "message": "TTS playback stopped"}
    except Exception as e:
        logger.error(f"Error stopping TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def listen_for_responses():
    """
    Background task that listens to response_text events from Redis.
    """
    global tts_processor
    
    logger.info("Started listening for responses...")
    redis_client = await get_redis_client()
    
    async def handle_response(data: dict):
        """Process incoming response text."""
        global tts_processor
        
        try:
            response = ResponseText(**data)
            logger.info(f"Received response: '{response.text}'")
            
            # Display text with clear formatting
            print("\n" + "="*60)
            print(f"🤖 ADRIAN: {response.text}")
            print("="*60)
            
            # Speak if requested
            if response.should_speak and tts_processor:
                print("🔊 Speaking now...\n")
                # Queue for async playback
                tts_processor.speak(response.text, blocking=False, interrupt=False)
            else:
                print("\n")
            
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
    Initialize text-to-speech engine using shared TTSProcessor.
    """
    global tts_processor
    
    logger.info("Initializing TTS engine...")
    
    try:
        # Create TTS config from settings
        config = TTSConfig(
            model_name=settings.tts_model_name,
            speaker=settings.tts_speaker,
            use_voice_cloning=settings.tts_use_voice_cloning,
            speed=settings.tts_speed,
            pitch_shift=settings.tts_pitch_shift,
            device=settings.tts_device,
            volume=settings.tts_volume
        )
        
        # Initialize TTS processor
        tts_processor = TTSProcessor(config)
        
        # Load model (this may take time, but runs in background)
        success = tts_processor.load_model()
        
        if success:
            # Set initial enabled state
            tts_processor.set_enabled(settings.tts_enabled)
            logger.info("✅ TTS engine initialized successfully")
        else:
            logger.error("❌ Failed to load TTS model")
            tts_processor = None
            
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}")
        tts_processor = None


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

