"""
Test script for complete voice input pipeline.
Tests: Hotword → VAD → STT → UtteranceEvent publishing

Usage:
    python services/io_service/test_pipeline.py

This will:
1. Start the IO service
2. Listen for "ADRIAN" hotword
3. Process your speech with STT
4. Publish UtteranceEvent to Redis
5. Display transcription and conversation state
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.logging_config import setup_logging
from shared.redis_client import get_redis_client
from shared.config import get_settings

# Import IO service components
from services.io_service.audio_utils import AudioStream
from services.io_service.conversation_manager import ConversationManager, ConversationState
from services.io_service.stt_utils import STTProcessor, get_stt_processor
from services.io_service.personality_error_handler import get_personality_handler
from services.io_service.vad_utils import VADState

logger = setup_logging("test-pipeline")
settings = get_settings()

# Global test components
audio_stream = None
conversation_manager = None
stt_processor = None
personality_handler = None
test_running = False


def on_hotword_detected():
    """Callback when ADRIAN hotword is detected."""
    try:
        logger.info("🎯 HOTWORD DETECTED: 'ADRIAN'")
        print("\n" + "="*60)
        print("✅ Hotword 'ADRIAN' detected!")
        print("🎤 Listening for your command...")
        print("="*60 + "\n")
        
        if conversation_manager:
            conversation_manager.start_conversation()
            print(f"📊 Conversation State: {conversation_manager.get_current_state().value}")
        
    except Exception as e:
        logger.error(f"Error in hotword callback: {e}")


def on_vad_state_change(is_speech: bool, vad_state_str: str):
    """Callback when VAD detects speech changes."""
    try:
        vad_state = VADState(vad_state_str)
        
        if conversation_manager and conversation_manager.is_in_conversation():
            if is_speech and vad_state == VADState.SPEECH:
                print("🎤 Speech detected - recording...")
                conversation_manager.current_state = ConversationState.LISTENING
                
            elif not is_speech and vad_state == VADState.SILENCE:
                print("🔇 Speech ended - processing with STT...")
                asyncio.create_task(_process_speech_segment())
                
    except Exception as e:
        logger.error(f"Error in VAD callback: {e}")


async def _process_speech_segment():
    """Process buffered speech with STT."""
    try:
        if not conversation_manager or not conversation_manager.is_in_conversation():
            return
        
        conversation_manager.current_state = ConversationState.PROCESSING
        print("⚙️ Processing audio with Whisper STT...")
        
        # Process audio buffer
        transcription, confidence, is_valid = await conversation_manager.process_audio_buffer()
        
        if is_valid and transcription:
            print(f"\n{'='*60}")
            print(f"✅ TRANSCRIPTION SUCCESS")
            print(f"{'='*60}")
            print(f"📝 Text: '{transcription}'")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"🔑 Conversation ID: {conversation_manager.current_context.conversation_id}")
            print(f"{'='*60}\n")
            
            # Simulate publishing to Redis (in real scenario this would publish)
            print("📤 Publishing UtteranceEvent to Redis...")
            await _publish_test_utterance(transcription, confidence)
            
            # Clear buffer
            conversation_manager.audio_buffer.clear()
            conversation_manager.is_buffering = True
            
        else:
            print(f"\n{'='*60}")
            print(f"⚠️ TRANSCRIPTION FAILED")
            print(f"{'='*60}")
            print(f"📝 Text: '{transcription}'")
            print(f"📊 Confidence: {confidence:.2%}")
            
            if personality_handler and conversation_manager.current_context:
                from services.io_service.stt_utils import STTError, AudioQuality
                error_response = personality_handler.get_error_response(
                    STTError.LOW_CONFIDENCE,
                    AudioQuality.UNKNOWN,
                    retry_count=0
                )
                print(f"🤖 ADRIAN: {error_response}")
            print(f"{'='*60}\n")
            
            conversation_manager.current_state = ConversationState.CONTINUOUS
        
        # Check timeout
        if not conversation_manager.should_continue_listening():
            print("⏱️ Conversation timeout - returning to idle state")
            conversation_manager.end_conversation()
        else:
            conversation_manager.current_state = ConversationState.CONTINUOUS
            print("💬 Continuing in conversational mode (say something else without 'ADRIAN')...")
        
    except Exception as e:
        logger.error(f"Error processing speech: {e}")
        print(f"❌ Error: {e}")
        if conversation_manager:
            conversation_manager.current_state = ConversationState.CONTINUOUS


async def _publish_test_utterance(transcription: str, confidence: float):
    """Test publishing utterance to Redis."""
    try:
        redis_client = await get_redis_client()
        
        import uuid
        from shared.schemas import UtteranceEvent
        
        utterance = UtteranceEvent(
            message_id=str(uuid.uuid4()),
            correlation_id=conversation_manager.current_context.conversation_id if conversation_manager.current_context else str(uuid.uuid4()),
            text=transcription,
            confidence=confidence,
            source="voice",
            user_id="test_user"
        )
        
        # Publish to Redis
        await redis_client.publish("utterances", utterance)
        
        print(f"✅ UtteranceEvent published to Redis channel 'utterances'")
        print(f"   Message ID: {utterance.message_id}")
        print(f"   Correlation ID: {utterance.correlation_id}")
        
    except Exception as e:
        logger.error(f"Error publishing utterance: {e}")
        print(f"❌ Failed to publish to Redis: {e}")


def on_audio_chunk(audio_data):
    """Callback for audio chunks."""
    if conversation_manager and conversation_manager.is_in_conversation():
        conversation_manager.add_audio_chunk(audio_data)


async def run_test():
    """Run the complete pipeline test."""
    global audio_stream, conversation_manager, stt_processor, personality_handler, test_running
    
    print("\n" + "="*60)
    print("🧪 ADRIAN Voice Input Pipeline Test")
    print("="*60)
    
    try:
        # Connect to Redis
        print("\n📡 Connecting to Redis...")
        redis_client = await get_redis_client()
        print("✅ Connected to Redis")
        
        # Initialize STT
        print(f"\n🎯 Loading Whisper model '{settings.whisper_model_name}'...")
        stt_processor = get_stt_processor()
        if not stt_processor.load_model():
            print("❌ Failed to load Whisper model")
            return
        print("✅ Whisper model loaded")
        
        # Initialize personality handler
        print("\n🤖 Initializing personality error handler...")
        personality_handler = get_personality_handler()
        print("✅ Personality handler ready")
        
        # Initialize conversation manager
        print(f"\n💬 Initializing conversation manager (timeout: {settings.conversation_timeout_seconds}s)...")
        conversation_manager = ConversationManager(
            conversation_timeout=settings.conversation_timeout_seconds,
            max_context_turns=settings.max_context_turns,
            confidence_threshold=settings.stt_confidence_threshold
        )
        print("✅ Conversation manager ready")
        
        # Initialize audio stream
        print(f"\n🎤 Starting audio stream with hotword detection...")
        audio_stream = AudioStream(
            callback=on_audio_chunk,
            enable_vad=True,
            vad_aggressiveness=settings.vad_aggressiveness,
            vad_callback=on_vad_state_change,
            enable_hotword=True,
            hotword_sensitivity=settings.hotword_sensitivity,
            hotword_callback=on_hotword_detected
        )
        
        audio_stream.start()
        print("✅ Audio stream started")
        
        print("\n" + "="*60)
        print("🎉 PIPELINE READY!")
        print("="*60)
        print("\n📋 Test Instructions:")
        print("   1. Say 'ADRIAN' (hotword)")
        print("   2. Wait for beep/confirmation")
        print("   3. Say your command (e.g., 'open chrome')")
        print("   4. Wait for transcription")
        print("   5. Continue conversation or wait 30s for timeout")
        print("\n💡 Tips:")
        print("   - Speak clearly and naturally")
        print("   - No need to repeat 'ADRIAN' during conversation")
        print("   - After 30s of silence, returns to idle")
        print("\n🛑 Press Ctrl+C to stop the test\n")
        
        test_running = True
        
        # Keep running until interrupted
        while test_running:
            await asyncio.sleep(1)
            
            # Display status every 10 seconds
            if conversation_manager and conversation_manager.is_in_conversation():
                summary = conversation_manager.get_conversation_summary()
                if summary:
                    # Show minimal status during conversation
                    pass
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {e}")
        print(f"\n❌ Test failed: {e}")
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        test_running = False
        
        if audio_stream:
            audio_stream.stop()
            print("✅ Audio stream stopped")
        
        if conversation_manager and conversation_manager.is_in_conversation():
            conversation_manager.end_conversation()
            print("✅ Conversation ended")
        
        if redis_client:
            await redis_client.disconnect()
            print("✅ Redis disconnected")
        
        print("\n👋 Test complete!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   ADRIAN Voice Input Pipeline - Integration Test")
    print("="*60)
    
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

