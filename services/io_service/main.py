"""
IO Service - Input Layer
Handles voice and text input, hotword detection, and speech-to-text.
Implements conversational mode with continuous listening and context awareness.
"""
import asyncio
import uuid
import signal
import sys
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import UtteranceEvent, HealthResponse, TTSPlaybackEvent
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Import audio and hotword utilities
from services.io_service.audio_utils import AudioStream, select_input_device, list_audio_devices
from services.io_service.hotword_utils import test_hotword_detection

# Import conversation management
from services.io_service.conversation_manager import ConversationManager, ConversationState, ConversationContext
from services.io_service.stt_utils import STTProcessor, AudioQuality, STTError, get_stt_processor
from services.io_service.personality_error_handler import PersonalityErrorHandler, get_personality_handler
from services.io_service.vad_utils import VADState
from services.io_service.pipeline_metrics import VoicePipelineMetrics

# Setup logging
logger = setup_logging("io-service")
settings = get_settings()

# Global components
audio_stream: Optional[AudioStream] = None
conversation_manager: Optional[ConversationManager] = None
stt_processor: Optional[STTProcessor] = None
personality_handler: Optional[PersonalityErrorHandler] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
pipeline_metrics: Optional[VoicePipelineMetrics] = None

# Flag to prevent duplicate STT processing
_stt_processing_lock = False

# Flag to prevent microphone feedback loop (pause VAD when ADRIAN is speaking, but keep hotword active)
_listening_for_responses = False
_mic_paused_for_tts = False
_tts_is_playing = False  # Track if TTS is currently playing
_last_hotword_timestamp: Optional[float] = None
_pending_resume_tasks: Dict[str, asyncio.Task] = {}


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
    global _tts_is_playing, pipeline_metrics, _last_hotword_timestamp
    
    try:
        # Check if TTS is currently playing - if so, interrupt it
        if _tts_is_playing:
            logger.info("🛑 Hotword detected during TTS - interrupting ADRIAN...")
            print("\n" + "="*60)
            print("🛑 INTERRUPTING ADRIAN - Hotword detected!")
            print("="*60 + "\n")
            
            # Interrupt TTS by calling Output Service API
            asyncio.run_coroutine_threadsafe(_interrupt_tts(), main_event_loop)
            
            # Resume microphone immediately
            if audio_stream and _mic_paused_for_tts:
                audio_stream.resume()
                _mic_paused_for_tts = False
                _tts_is_playing = False
            
            return
        
        # Normal hotword detection (when TTS is not playing)
        print("\n" + "="*60)
        print("🎯 ADRIAN HOTWORD DETECTED! Listening...")
        print("="*60 + "\n")
        
        logger.info("🎯 ADRIAN hotword detected! Starting conversation...")
        _last_hotword_timestamp = time.perf_counter()
        
        # Play a beep sound for feedback (configurable)
        if settings.activation_beep_enabled:
            try:
                import platform
                if platform.system() == "Windows":
                    import winsound
                    winsound.Beep(settings.activation_beep_frequency, settings.activation_beep_duration_ms)
                else:
                    import os
                    os.system('echo -e "\a"')
            except Exception as e:
                logger.debug(f"Could not play activation beep: {e}")
        
        # Start conversation in conversation manager
        if conversation_manager:
            start_ts = time.time()
            conversation_id = conversation_manager.start_conversation(hotword_timestamp=start_ts)
            if pipeline_metrics and settings.pipeline_metrics_enabled:
                pipeline_metrics.record_stage(conversation_id, "hotword")
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
    global pipeline_metrics
    try:
        if not conversation_manager or not conversation_manager.is_in_conversation():
            return
        
        print("\n🎤 Processing speech...\n")
        
        # Update state to processing
        conversation_manager.current_state = ConversationState.PROCESSING
        
        # Process audio buffer with STT
        transcription, confidence, is_valid = await conversation_manager.process_audio_buffer()
        
        context = conversation_manager.current_context
        if is_valid and transcription:
            print(f"💬 You said: '{transcription}' (confidence: {confidence:.2f})")
            print("🤖 Waiting for ADRIAN's response...\n")
            
            if pipeline_metrics and context:
                pipeline_metrics.record_stage(
                    context.conversation_id,
                    "stt_completed",
                    transcript=transcription,
                    confidence=confidence,
                    retries=context.retry_count
                )
                context.retry_count = 0
            
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
            if context:
                context.retry_count += 1
                if pipeline_metrics:
                    pipeline_metrics.record_stage(context.conversation_id, "retry")
                    pipeline_metrics.record_error(context.conversation_id, "unclear_speech")
                if (settings.conversation_retry_limit > 0 and 
                        context.retry_count >= settings.conversation_retry_limit):
                    if settings.error_beep_enabled:
                        try:
                            import platform
                            if platform.system() == "Windows":
                                import winsound
                                winsound.Beep(settings.error_beep_frequency, settings.error_beep_duration_ms)
                            else:
                                import os
                                os.system('echo -e "\a"')
                        except Exception as beep_err:
                            logger.debug(f"Could not play error beep: {beep_err}")
                    apology = "I'm sorry, I still couldn't understand. Let's try again in a moment."
                    await _publish_error_response(apology, context)
                    conversation_manager.end_conversation()
            
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
# Echo Prevention: Pause microphone when ADRIAN is speaking
# =============================================================================

async def _interrupt_tts():
    """Interrupt TTS playback by calling Output Service API."""
    try:
        import httpx
        from shared.config import get_settings
        settings = get_settings()
        
        # Call Output Service to stop TTS
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{settings.output_service_port}/tts/stop",
                json={}
            )
            if response.status_code == 200:
                logger.info("✅ TTS interrupted successfully")
            else:
                logger.warning(f"Failed to interrupt TTS: {response.status_code}")
    except Exception as e:
        logger.error(f"Error interrupting TTS: {e}")
        # Fallback: try to stop playback directly via stop_playback
        # This requires access to TTS processor, which we don't have in IO Service
        # So we rely on the API call


async def _handle_response_messages():
    """
    Subscribe to 'responses' channel to know when ADRIAN is speaking.
    Pause VAD/callback during TTS (but keep hotword active for interruption).
    """
    global audio_stream, _mic_paused_for_tts, _tts_is_playing, pipeline_metrics, _pending_resume_tasks
    
    logger.info("Starting response listener for echo prevention (hotword interruption enabled)...")
    redis_client = await get_redis_client()
    
    async def handle_response(data: dict):
        """Handle incoming response - pause VAD during TTS, but keep hotword active."""
        global audio_stream, _mic_paused_for_tts, _tts_is_playing
        
        try:
            from shared.schemas import ResponseText
            response = ResponseText(**data)
            
            # Only pause if response should be spoken
            if response.should_speak and audio_stream and audio_stream.is_running:
                correlation_id = response.correlation_id
                if pipeline_metrics and settings.pipeline_metrics_enabled:
                    # Get processing metrics from response metadata if available
                    processing_metrics = getattr(response, 'metadata', {}) or {}
                    pipeline_metrics.record_stage(
                        correlation_id,
                        "response_received",
                        response_text=response.text,
                        intent=processing_metrics.get("intent"),
                        intent_confidence=processing_metrics.get("intent_confidence"),
                        route_type=processing_metrics.get("route_type"),
                        route_reason=processing_metrics.get("route_reason"),
                        intent_classification_latency_ms=processing_metrics.get("intent_classification_latency_ms"),
                        routing_decision_latency_ms=processing_metrics.get("routing_decision_latency_ms"),
                        llm_latency_ms=processing_metrics.get("llm_latency_ms"),
                        response_generation_latency_ms=processing_metrics.get("response_generation_latency_ms"),
                    )

                word_count = len(response.text.split())
                estimated_duration = max((word_count / 2.5) + 0.5, 2.0)

                logger.info(f"🔇 Pausing VAD (estimate {estimated_duration:.1f}s) for TTS: '{response.text[:50]}...'")
                logger.info("   💡 Hotword detection still active - say 'ADRIAN' to interrupt!")
                print(f"\n[ECHO PREVENTION] Pausing normal speech recognition...")
                print(f"   Say 'ADRIAN' to interrupt if needed!\n")
                
                _mic_paused_for_tts = True
                _tts_is_playing = True
                audio_stream.pause()  # Pause VAD and callback (hotword still active)
                # Fallback in case we miss TTS end event
                if correlation_id:
                    resume_task = asyncio.create_task(_resume_after_timeout(correlation_id, estimated_duration + 1.0))
                    _pending_resume_tasks[correlation_id] = resume_task
                
        except Exception as e:
            logger.error(f"Error handling response for echo prevention: {e}")
            # Resume microphone on error
            if audio_stream and _mic_paused_for_tts:
                audio_stream.resume()
                _mic_paused_for_tts = False
                _tts_is_playing = False
    
    # Subscribe to responses channel
    await redis_client.subscribe("responses", handle_response)


async def _resume_after_timeout(correlation_id: str, delay_seconds: float):
    """Fallback resume if TTS completion event is missed."""
    try:
        await asyncio.sleep(delay_seconds)
        logger.warning(f"TTS end event not received for {correlation_id} - auto-resuming microphone")
        _resume_microphone(correlation_id, reason="fallback_timer")
    except asyncio.CancelledError:
        logger.debug(f"Fallback resume cancelled for {correlation_id}")
        raise


def _resume_microphone(correlation_id: Optional[str], reason: str = "tts_event"):
    """Resume microphone input after TTS finishes or is interrupted."""
    global _mic_paused_for_tts, _tts_is_playing

    task = _pending_resume_tasks.pop(correlation_id, None)
    if task and not task.cancelled():
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if task is not current_task:
            task.cancel()

    if audio_stream and audio_stream.is_running and _mic_paused_for_tts:
        audio_stream.resume()
        logger.info(f"✅ VAD resumed ({reason})")
        print("[ECHO PREVENTION] Normal speech recognition resumed\n")
    _mic_paused_for_tts = False
    _tts_is_playing = False


async def _handle_tts_events():
    """Listen for playback events to resume microphone accurately."""
    global _tts_is_playing

    logger.info("Starting TTS event listener for echo prevention...")
    redis_client = await get_redis_client()

    async def handle_event(data: dict):
        global _tts_is_playing
        try:
            event = TTSPlaybackEvent(**data)
            correlation_id = event.conversation_id or event.correlation_id

            if event.event == "start":
                _tts_is_playing = True
                if pipeline_metrics and settings.pipeline_metrics_enabled and correlation_id:
                    pipeline_metrics.record_stage(correlation_id, "tts_started")
            elif event.event == "end":
                if pipeline_metrics and settings.pipeline_metrics_enabled and correlation_id:
                    pipeline_metrics.record_stage(correlation_id, "tts_completed")
                _resume_microphone(correlation_id, reason="tts_completed")
            elif event.event == "interrupted":
                if pipeline_metrics and settings.pipeline_metrics_enabled and correlation_id:
                    pipeline_metrics.record_stage(correlation_id, "interrupted")
                _resume_microphone(correlation_id, reason="tts_interrupted")
            elif event.event == "error":
                if pipeline_metrics and settings.pipeline_metrics_enabled and correlation_id:
                    pipeline_metrics.record_error(correlation_id, "tts_error")
                _resume_microphone(correlation_id, reason="tts_error")
        except Exception as exc:
            logger.error(f"Error processing TTS event: {exc}")

    await redis_client.subscribe("tts_events", handle_event)


# =============================================================================
# TTS is handled by Output Service, not IO Service
# IO Service is input-only (microphone, STT, hotword detection)
# All responses are published to Redis and handled by Output Service
#


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global audio_stream, conversation_manager, stt_processor, personality_handler, main_event_loop, _listening_for_responses
    global pipeline_metrics
    
    # Store the main event loop for callbacks
    main_event_loop = asyncio.get_event_loop()
    
    logger.info("🚀 Starting IO Service...")
    if settings.pipeline_metrics_enabled:
        pipeline_metrics = VoicePipelineMetrics(settings.pipeline_latency_target_ms)
        logger.info(f"Pipeline metrics enabled (target: {settings.pipeline_latency_target_ms}ms)")
    else:
        pipeline_metrics = None
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("✅ Connected to Redis")
    
    # Start listening for responses (to pause mic during TTS)
    _listening_for_responses = True
    asyncio.create_task(_handle_response_messages())
    asyncio.create_task(_handle_tts_events())
    logger.info("✅ Response listener started (echo prevention enabled)")
    
    # Initialize STT processor with configuration
    try:
        logger.info(f"Initializing Whisper STT processor (model: {settings.whisper_model_name}, device: {settings.whisper_device})...")
        # Use absolute import (already imported at top)
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
    
    # TTS is handled by Output Service, not IO Service
    # IO Service only handles input (microphone, STT, hotword detection)
    
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
        mic_device_index = settings.microphone_device_index
        if mic_device_index is None:
            try:
                mic_device_index = select_input_device(settings.microphone_device_name)
            except Exception as device_err:
                logger.warning(f"Failed to resolve microphone device: {device_err}")
                mic_device_index = None

        if mic_device_index is not None:
            logger.info(f"Using microphone device index: {mic_device_index}")

        audio_stream = AudioStream(
            callback=on_audio_chunk,
            device=mic_device_index,
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
        print("\n" + "="*60)
        print("✅ IO Service ready - Say 'ADRIAN' to start!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to initialize audio stream: {e}")
        audio_stream = None
    
    logger.info("🎉 IO Service startup complete - ready for voice input!")
    logger.info("📢 Note: TTS output is handled by Output Service (port 8006)")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down IO Service...")
    
    # Stop listening for responses (already declared global at function start)
    _listening_for_responses = False
    
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
    conversation_status = "active" if (conversation_manager and conversation_manager.is_in_conversation()) else "idle"
    
    return HealthResponse(
        service_name="io-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "stt_engine": stt_status,
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


# =============================================================================
# Pipeline Metrics & Audio Device Introspection
# =============================================================================


@app.get("/status/pipeline")
async def get_pipeline_status():
    """Return pipeline latency metrics."""
    if not settings.pipeline_metrics_enabled or pipeline_metrics is None:
        return {
            "status": "disabled",
            "message": "Pipeline metrics tracking disabled"
        }
    return {
        "status": "ok",
        "metrics": pipeline_metrics.get_summary()
    }


@app.get("/audio/devices")
async def list_devices():
    """List available audio input/output devices."""
    try:
        devices = list_audio_devices()
        return {
            "status": "ok",
            "count": len(devices),
            "devices": devices
        }
    except Exception as exc:
        logger.error(f"Failed to list audio devices: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# TTS status endpoint removed - TTS is handled by Output Service
# Use GET /tts/status on Output Service (port 8006) instead




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

