"""
IO Service - Input Layer
Handles voice and text input, hotword detection, and speech-to-text.
Implements conversational mode with continuous listening and context awareness.
"""
import asyncio
import uuid
import signal
import sys
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import UtteranceEvent, HealthResponse
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Import audio and hotword utilities
from services.io_service.audio_utils import AudioStream
from services.io_service.hotword_utils import test_hotword_detection

# Import conversation management
from services.io_service.conversation_manager import ConversationManager, ConversationState, ConversationContext
from services.io_service.stt_utils import STTProcessor, AudioQuality, STTError, get_stt_processor
from services.io_service.personality_error_handler import PersonalityErrorHandler, get_personality_handler
from services.io_service.vad_utils import VADState
from services.io_service.tts_utils import TTSProcessor, TTSConfig

# Setup logging
logger = setup_logging("io-service")
settings = get_settings()

# Global components
audio_stream: Optional[AudioStream] = None
conversation_manager: Optional[ConversationManager] = None
stt_processor: Optional[STTProcessor] = None
personality_handler: Optional[PersonalityErrorHandler] = None
tts_processor: Optional[TTSProcessor] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Flag to prevent duplicate STT processing
_stt_processing_lock = False


# Request models
class TextInputRequest(BaseModel):
    """Request model for text input."""
    text: str
    user_id: str = "default_user"


# =============================================================================
# Signal Handlers for Clean Shutdown
# =============================================================================

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("\n🛑 Received interrupt signal (Ctrl+C), shutting down...")
    
    # Stop audio stream immediately
    if audio_stream:
        try:
            audio_stream.stop()
            logger.info("Audio stream stopped")
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
    
    # Exit
    sys.exit(0)


# =============================================================================
# Callbacks for Audio Pipeline
# =============================================================================

def on_hotword_detected():
    """Called when 'ADRIAN' hotword is detected."""
    try:
        logger.info("🎯 ADRIAN hotword detected! Starting conversation...")
        
        # Start conversation in conversation manager
        if conversation_manager:
            conversation_manager.start_conversation()
            logger.info(f"📊 Conversation state: {conversation_manager.get_current_state().value}")
        
    except Exception as e:
        logger.error(f"Error in hotword callback: {e}")


def on_vad_state_change(is_speech: bool, vad_state_str: str):
    """Called when VAD detects speech start/end."""
    global _stt_processing_lock
    
    try:
        vad_state = VADState(vad_state_str)
        
        # Only process if in an active conversation
        if conversation_manager and conversation_manager.is_in_conversation():
            
            if is_speech and vad_state == VADState.SPEECH:
                # Speech started - conversation manager is already buffering
                logger.debug("🎤 Speech started (VAD)")
                conversation_manager.current_state = ConversationState.LISTENING
                # Reset the lock when new speech starts
                _stt_processing_lock = False
                
            elif not is_speech and vad_state == VADState.SILENCE:
                # Speech ended - trigger STT processing ONLY ONCE
                if not _stt_processing_lock:
                    _stt_processing_lock = True  # Lock immediately to prevent duplicates
                    logger.debug("🔇 Speech ended (VAD) - triggering STT")
                    # Schedule the async task using the main event loop
                    if main_event_loop and main_event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(_process_speech_segment(), main_event_loop)
                    else:
                        logger.warning("Cannot process speech: event loop not available")
                
    except Exception as e:
        logger.error(f"Error in VAD callback: {e}")


async def _process_speech_segment():
    """Process the buffered speech segment with STT."""
    try:
        if not conversation_manager or not conversation_manager.is_in_conversation():
            return
        
        # Update state to processing
        conversation_manager.current_state = ConversationState.PROCESSING
        
        # Process audio buffer with STT
        transcription, confidence, is_valid = await conversation_manager.process_audio_buffer()
        
        if is_valid and transcription:
            # Publish UtteranceEvent to Redis
            await _publish_utterance(transcription, confidence, conversation_manager.current_context)
            
            # Clear audio buffer for next utterance
            conversation_manager.audio_buffer.clear()
            conversation_manager.is_buffering = True
            
        else:
            # Generate personality error response when transcription is invalid/unclear
            if personality_handler and conversation_manager.current_context:
                error_response = personality_handler.get_error_response(
                    STTError.UNCLEAR_SPEECH,
                    AudioQuality.UNCLEAR,
                    retry_count=0
                )
                logger.info(f"🤖 ADRIAN says: {error_response}")
                
                # Optionally publish error response so ADRIAN can speak it
                await _publish_error_response(error_response, conversation_manager.current_context)
            
            # Continue listening
            conversation_manager.current_state = ConversationState.CONTINUOUS
        
        # Check if we should continue listening or timeout
        if not conversation_manager.should_continue_listening():
            logger.info("⏱️ Conversation timeout - returning to idle")
            conversation_manager.end_conversation()
        else:
            # Continue in conversational mode
            conversation_manager.current_state = ConversationState.CONTINUOUS
            logger.debug("💬 Continuing in conversational mode")
        
    except Exception as e:
        logger.error(f"Error processing speech segment: {e}")
        if conversation_manager:
            conversation_manager.current_state = ConversationState.CONTINUOUS


async def _publish_utterance(transcription: str, confidence: float, context: Optional[ConversationContext]):
    """Publish UtteranceEvent to Redis."""
    try:
        redis_client = await get_redis_client()
        
        # Create utterance event
        utterance = UtteranceEvent(
            message_id=str(uuid.uuid4()),
            correlation_id=context.conversation_id if context else str(uuid.uuid4()),
            text=transcription,
            confidence=confidence,
            source="voice",
            user_id=context.user_id if context else "default_user"
        )
        
        # Publish to Redis
        await redis_client.publish("utterances", utterance)
        
        logger.info(f"✅ Utterance published: '{transcription}' (confidence: {confidence:.2f})")
        
    except Exception as e:
        logger.error(f"Error publishing utterance: {e}")


async def _publish_error_response(error_text: str, context: Optional[ConversationContext]):
    """Publish error response as ResponseText event."""
    try:
        redis_client = await get_redis_client()
        
        # Import ResponseText schema
        from shared.schemas import ResponseText
        
        # Create response event
        response = ResponseText(
            message_id=str(uuid.uuid4()),
            correlation_id=context.conversation_id if context else str(uuid.uuid4()),
            text=error_text,
            should_speak=True,
            emotion="apologetic"
        )
        
        # Publish to Redis for output service
        await redis_client.publish("responses", response)
        
        logger.info(f"📤 Error response published: '{error_text}'")
        
    except Exception as e:
        logger.error(f"Error publishing error response: {e}")


def on_audio_chunk(audio_data):
    """Called for each audio chunk from the stream."""
    # Add to conversation manager buffer if in conversation
    if conversation_manager and conversation_manager.is_in_conversation():
        conversation_manager.add_audio_chunk(audio_data)


# =============================================================================
# TTS Response Handler
# =============================================================================

async def _handle_response_messages():
    """Subscribe to Redis responses channel and speak them via TTS."""
    try:
        redis_client = await get_redis_client()
        
        logger.info("🔊 Starting TTS response handler...")
        
        # Import ResponseText schema
        from shared.schemas import ResponseText
        
        # Define message handler
        async def handle_response(message_data):
            try:
                # Parse response
                if isinstance(message_data, dict):
                    response = ResponseText(**message_data)
                else:
                    response = message_data
                
                # Check if we should speak this response
                if response.should_speak and tts_processor:
                    logger.info(f"🗣️ Speaking: '{response.text[:50]}...'")
                    
                    # Speak the response (non-blocking, queued)
                    tts_processor.speak(response.text, blocking=False)
                else:
                    logger.debug(f"Skipping TTS for: '{response.text[:50]}...'")
                    
            except Exception as e:
                logger.error(f"Error processing response message: {e}")
        
        # Subscribe to responses channel
        await redis_client.subscribe("responses", handle_response)
        
    except Exception as e:
        logger.error(f"TTS response handler error: {e}")


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global audio_stream, conversation_manager, stt_processor, personality_handler, tts_processor, main_event_loop
    
    # Store the main event loop for callbacks
    main_event_loop = asyncio.get_event_loop()
    
    logger.info("🚀 Starting IO Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("✅ Connected to Redis")
    
    # Initialize STT processor with configuration
    try:
        logger.info(f"Initializing Whisper STT processor (model: {settings.whisper_model_name}, device: {settings.whisper_device})...")
        from .stt_utils import STTProcessor
        stt_processor = STTProcessor(
            model_name=settings.whisper_model_name,
            device=settings.whisper_device
        )
        if stt_processor.load_model():
            logger.info("✅ Whisper model loaded successfully")
        else:
            logger.error("❌ Failed to load Whisper model")
            stt_processor = None
    except Exception as e:
        logger.error(f"Failed to initialize STT processor: {e}")
        stt_processor = None
    
    # Initialize personality error handler
    try:
        logger.info("Initializing personality error handler...")
        personality_handler = get_personality_handler()
        logger.info("✅ Personality error handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize personality handler: {e}")
        personality_handler = None
    
    # Initialize TTS processor with configuration
    try:
        logger.info(f"Initializing TTS processor (model: {settings.tts_model_name}, voice cloning: {settings.tts_use_voice_cloning}, speed: {settings.tts_speed}, pitch: {settings.tts_pitch_shift})...")
        tts_config = TTSConfig(
            model_name=settings.tts_model_name,
            speaker=settings.tts_speaker,
            use_voice_cloning=getattr(settings, 'tts_use_voice_cloning', False),
            speed=settings.tts_speed,
            pitch_shift=settings.tts_pitch_shift,
            device=settings.tts_device
        )
        tts_processor = TTSProcessor(config=tts_config)
        if tts_processor.load_model():
            logger.info("✅ TTS model loaded successfully")
        else:
            logger.error("❌ Failed to load TTS model")
            tts_processor = None
    except Exception as e:
        logger.error(f"Failed to initialize TTS processor: {e}")
        tts_processor = None
    
    # Initialize conversation manager with configuration
    try:
        logger.info(f"Initializing conversation manager (timeout: {settings.conversation_timeout_seconds}s, context_turns: {settings.max_context_turns})...")
        conversation_manager = ConversationManager(
            conversation_timeout=settings.conversation_timeout_seconds,
            max_context_turns=settings.max_context_turns,
            confidence_threshold=settings.stt_confidence_threshold,
            stt_processor=stt_processor  # Pass the loaded STT processor
        )
        logger.info("✅ Conversation manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize conversation manager: {e}")
        conversation_manager = None
    
    # Initialize audio stream with full pipeline integration and configuration
    try:
        logger.info(f"Initializing audio stream (VAD: {settings.vad_aggressiveness}, hotword_sens: {settings.hotword_sensitivity})...")
        audio_stream = AudioStream(
            callback=on_audio_chunk,
            enable_vad=True,
            vad_aggressiveness=settings.vad_aggressiveness,
            vad_callback=on_vad_state_change,
            enable_hotword=True,
            hotword_sensitivity=settings.hotword_sensitivity,
            hotword_callback=on_hotword_detected
        )
        
        # Auto-start audio stream for continuous listening
        audio_stream.start()
        logger.info("✅ Audio stream started - listening for 'ADRIAN'...")
        
    except Exception as e:
        logger.error(f"Failed to initialize audio stream: {e}")
        audio_stream = None
    
    # Start TTS response handler in background
    tts_task = None
    if tts_processor:
        tts_task = asyncio.create_task(_handle_response_messages())
        logger.info("✅ TTS response handler started")
    
    logger.info("🎉 IO Service startup complete - ready for voice input!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down IO Service...")
    
    # Cancel TTS response handler
    if tts_task:
        tts_task.cancel()
        try:
            await tts_task
        except asyncio.CancelledError:
            pass
    
    # Stop TTS processor
    if tts_processor:
        tts_processor.shutdown()
        logger.info("TTS processor stopped")
    
    # Stop audio stream
    if audio_stream:
        audio_stream.stop()
        logger.info("Audio stream stopped")
    
    # End any active conversation
    if conversation_manager and conversation_manager.is_in_conversation():
        conversation_manager.end_conversation()
    
    await redis_client.disconnect()
    logger.info("👋 IO Service shutdown complete")


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
    # Check component statuses
    hotword_status = "disabled"
    if audio_stream and audio_stream.is_hotword_enabled():
        status = audio_stream.get_hotword_status()
        hotword_status = "enabled" if status.get("is_running", False) else "stopped"
    
    stt_status = "loaded" if (stt_processor and stt_processor.is_model_loaded) else "not_loaded"
    tts_status = "loaded" if (tts_processor and tts_processor.is_model_loaded) else "not_loaded"
    conversation_status = "active" if (conversation_manager and conversation_manager.is_in_conversation()) else "idle"
    
    return HealthResponse(
        service_name="io-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "stt_engine": stt_status,
            "tts_engine": tts_status,
            "microphone": "available" if audio_stream else "unavailable",
            "hotword_detection": hotword_status,
            "conversation_manager": conversation_status,
            "personality_handler": "loaded" if personality_handler else "not_loaded"
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


@app.get("/status/conversation")
async def get_conversation_status():
    """
    Get current conversation status and context.
    """
    global conversation_manager
    
    try:
        if conversation_manager is None:
            return {
                "status": "not_initialized",
                "message": "Conversation manager not initialized"
            }
        
        # Get conversation summary
        summary = conversation_manager.get_conversation_summary()
        
        # Get performance stats
        stats = conversation_manager.get_performance_stats()
        
        return {
            "status": "ok",
            "current_state": conversation_manager.get_current_state().value,
            "is_in_conversation": conversation_manager.is_in_conversation(),
            "conversation_summary": summary,
            "performance_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation status: {e}")
        return {
            "status": "error",
            "message": f"Error getting status: {str(e)}"
        }


@app.get("/status/stt")
async def get_stt_status():
    """
    Get STT processor status and performance metrics.
    """
    global stt_processor
    
    try:
        if stt_processor is None:
            return {
                "status": "not_initialized",
                "message": "STT processor not initialized"
            }
        
        # Get performance stats
        stats = stt_processor.get_performance_stats()
        
        return {
            "status": "ok",
            "model_loaded": stt_processor.is_model_loaded,
            "model_name": stt_processor.model_name,
            "device": stt_processor.device,
            "performance_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting STT status: {e}")
        return {
            "status": "error",
            "message": f"Error getting status: {str(e)}"
        }


@app.get("/status/tts")
async def get_tts_status():
    """
    Get TTS processor status and performance metrics.
    """
    global tts_processor
    
    try:
        if tts_processor is None:
            return {
                "status": "not_initialized",
                "message": "TTS processor not initialized"
            }
        
        # Get performance stats
        stats = tts_processor.get_performance_stats()
        
        return {
            "status": "ok",
            "model_loaded": tts_processor.is_model_loaded,
            "model_name": tts_processor.config.model_name,
            "speed": tts_processor.config.speed,
            "performance_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting TTS status: {e}")
        return {
            "status": "error",
            "message": f"Error getting status: {str(e)}"
        }




if __name__ == "__main__":
    import uvicorn
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.io_service_port,
        log_level=settings.log_level.lower()
    )

