"""
Output Service - TTS and UI Layer
Handles text-to-speech, visual output, and system notifications.
"""
import asyncio
import sys
import uuid
from collections import deque
from typing import Optional, Deque
from pathlib import Path
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.config import get_settings
from shared.schemas import ResponseText, ActionResult, HealthResponse, TTSPlaybackEvent
from shared.redis_client import get_redis_client, RedisClient
from shared.logging_config import setup_logging
from shared.tts_processor import TTSProcessor, TTSConfig
from services.output_service.tray_indicator import TrayIndicator

# Setup logging
logger = setup_logging("output-service")
settings = get_settings()

# Global state
tts_processor: TTSProcessor = None
redis_client: Optional[RedisClient] = None
playback_event_queue: Optional[asyncio.Queue] = None
playback_event_task: Optional[asyncio.Task] = None
pending_playback: Deque[dict] = deque()
current_playback: Optional[dict] = None
tray_indicator: Optional[TrayIndicator] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


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


class VoiceRequest(BaseModel):
    """Request to change active TTS voice."""
    speaker: str


class OutputDeviceRequest(BaseModel):
    """Request to change output audio device."""
    index: Optional[int] = None
    name: Optional[str] = None


# Background task flags
_listening_responses = False
_listening_results = False


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _listening_responses, _listening_results, tts_processor, redis_client
    global playback_event_queue, playback_event_task, tray_indicator, main_event_loop
    logger.info("Starting Output Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    main_event_loop = asyncio.get_event_loop()
    playback_event_queue = asyncio.Queue()
    if settings.tray_indicator_enabled:
        tray_indicator_local = TrayIndicator()
        tray_indicator_local.start()
        tray_indicator = tray_indicator_local
        if tray_indicator:
            tray_indicator.set_state("idle")
    
    # Initialize TTS processor
    await initialize_tts()
    
    # Start background listeners
    _listening_responses = True
    _listening_results = True
    asyncio.create_task(listen_for_responses())
    asyncio.create_task(listen_for_action_results())
    playback_event_task = asyncio.create_task(process_playback_events())
    
    print("\n" + "="*60)
    print("✅ Output Service ready - TTS enabled!")
    print("="*60 + "\n")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Output Service...")
    _listening_responses = False
    _listening_results = False
    if playback_event_task:
        playback_event_task.cancel()
        with suppress(asyncio.CancelledError):
            await playback_event_task
    
    # Shutdown TTS processor
    if tts_processor:
        tts_processor.shutdown()
    
    if tray_indicator:
        tray_indicator.set_state("idle")
        tray_indicator.stop()
    if redis_client:
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


@app.post("/tts/voice")
async def set_voice(request: VoiceRequest):
    """Change the active speaker voice."""
    global tts_processor

    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    available = tts_processor.list_available_voices()
    if available and request.speaker not in available:
        raise HTTPException(status_code=400, detail=f"Speaker '{request.speaker}' not available")

    tts_processor.set_voice(request.speaker)
    return {"status": "success", "speaker": request.speaker}


@app.get("/tts/voices")
async def list_voices():
    """List available TTS voices."""
    global tts_processor

    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    return {"voices": tts_processor.list_available_voices()}


@app.post("/tts/output-device")
async def set_output_device(request: OutputDeviceRequest):
    """Update the audio output device by name or index."""
    global tts_processor

    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    if request.index is None and not request.name:
        raise HTTPException(status_code=400, detail="Provide either 'index' or 'name'")

    tts_processor.set_output_device(index=request.index, name=request.name)
    stats = tts_processor.get_performance_stats()
    return {
        "status": "success",
        "output_device_index": stats.get("output_device_index"),
        "hint": "Restart playback for changes to take effect"
    }


@app.get("/tts/output-device")
async def get_output_device():
    """Return current output device configuration."""
    global tts_processor

    if not tts_processor:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    stats = tts_processor.get_performance_stats()
    return {
        "output_device_index": stats.get("output_device_index"),
        "available_voices": tts_processor.list_available_voices()
    }


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
    redis_client_local = await get_redis_client()
    
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
                pending_payload = {
                    "correlation_id": response.correlation_id,
                    "text": response.text
                }
                pending_playback.append(pending_payload)
                if tray_indicator:
                    tray_indicator.set_state("speaking")
                # Queue for async playback
                if not tts_processor.speak(response.text, blocking=False, interrupt=False):
                    # Remove pending payload if queueing failed
                    if pending_playback and pending_playback[-1] == pending_payload:
                        pending_playback.pop()
            else:
                print("\n")
                if tray_indicator:
                    tray_indicator.set_state("idle")
            
        except Exception as e:
            logger.error(f"Error handling response: {e}")
    
    # Subscribe and listen
    await redis_client_local.subscribe("responses", handle_response)


async def listen_for_action_results():
    """
    Background task that listens to action_result events from Redis.
    """
    logger.info("Started listening for action results...")
    redis_client_local = await get_redis_client()
    
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
    await redis_client_local.subscribe("action_results", handle_action_result)


def _enqueue_playback_event(event: str, payload: dict):
    if playback_event_queue is None or main_event_loop is None:
        return
    try:
        main_event_loop.call_soon_threadsafe(playback_event_queue.put_nowait, (event, payload))
    except RuntimeError as exc:
        logger.debug(f"Failed to enqueue playback event: {exc}")


def playback_event_handler(event: str, payload: dict):
    """Handle playback events from TTS processor (executed in worker thread)."""
    global current_playback, pending_playback

    combined = dict(payload)

    if event == "start":
        metadata = pending_playback.popleft() if pending_playback else {}
        current_playback = metadata or None
        combined = {**metadata, **payload}
    elif event in {"end", "error", "interrupted"}:
        metadata = current_playback or {}
        combined = {**metadata, **payload}
        if event in {"end", "interrupted", "error"}:
            current_playback = None

    _enqueue_playback_event(event, combined)


async def process_playback_events():
    """Publish playback events to Redis and update tray indicator."""
    while True:
        try:
            if playback_event_queue is None:
                await asyncio.sleep(0.1)
                continue

            event, payload = await playback_event_queue.get()
            try:
                correlation_id = payload.get("correlation_id") or str(uuid.uuid4())
                message = TTSPlaybackEvent(
                    message_id=str(uuid.uuid4()),
                    correlation_id=correlation_id,
                    event=event,
                    conversation_id=payload.get("correlation_id"),
                    text=payload.get("text"),
                    duration_ms=payload.get("duration_ms"),
                    metadata={k: v for k, v in payload.items() if k not in {"text", "duration_ms", "correlation_id"}}
                )
                if redis_client:
                    await redis_client.publish("tts_events", message)

                if tray_indicator:
                    if event == "start":
                        tray_indicator.set_state("speaking")
                    elif event == "end":
                        tray_indicator.set_state("idle")
                    elif event == "error":
                        tray_indicator.set_state("error")
                    elif event == "interrupted":
                        tray_indicator.set_state("idle")
            finally:
                if playback_event_queue:
                    playback_event_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Playback event processing error: {exc}")
            await asyncio.sleep(0.1)


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
            volume=settings.tts_volume,
            output_device_index=settings.speaker_device_index,
            output_device_name=settings.speaker_device_name
        )
        
        # Initialize TTS processor
        tts_processor = TTSProcessor(config)
        
        # Load model (this may take time, but runs in background)
        success = tts_processor.load_model()
        
        if success:
            # Set initial enabled state
            tts_processor.set_enabled(settings.tts_enabled)
            tts_processor.set_playback_event_handler(playback_event_handler)
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

