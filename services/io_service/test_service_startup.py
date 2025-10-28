"""
Quick test to verify IO Service starts correctly with all components.
Tests startup sequence without requiring actual voice input.

Usage:
    python services/io_service/test_service_startup.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.logging_config import setup_logging
from shared.config import get_settings

logger = setup_logging("test-startup")
settings = get_settings()


async def test_startup():
    """Test that all components initialize correctly."""
    print("\n" + "="*70)
    print("   ADRIAN IO Service - Startup Test")
    print("="*70 + "\n")
    
    errors = []
    
    # Test 1: Redis Connection
    print("🧪 Test 1: Redis Connection")
    try:
        from shared.redis_client import get_redis_client
        redis_client = await get_redis_client()
        print("   ✅ Redis connected")
    except Exception as e:
        print(f"   ❌ Redis connection failed: {e}")
        errors.append(("Redis", e))
    
    # Test 2: STT Processor
    print("\n🧪 Test 2: Whisper STT Processor")
    try:
        from services.io_service.stt_utils import STTProcessor
        stt_processor = STTProcessor(
            model_name=settings.whisper_model_name,
            device=settings.whisper_device
        )
        if stt_processor.load_model():
            print(f"   ✅ Whisper model '{settings.whisper_model_name}' loaded")
            print(f"   📊 Device: {settings.whisper_device}")
        else:
            print(f"   ❌ Failed to load Whisper model")
            errors.append(("Whisper", "Model load failed"))
    except Exception as e:
        print(f"   ❌ STT processor initialization failed: {e}")
        errors.append(("STT Processor", e))
    
    # Test 3: Personality Handler
    print("\n🧪 Test 3: Personality Error Handler")
    try:
        from services.io_service.personality_error_handler import get_personality_handler
        personality_handler = get_personality_handler()
        print("   ✅ Personality handler initialized")
    except Exception as e:
        print(f"   ❌ Personality handler failed: {e}")
        errors.append(("Personality Handler", e))
    
    # Test 4: Conversation Manager
    print("\n🧪 Test 4: Conversation Manager")
    try:
        from services.io_service.conversation_manager import ConversationManager
        conversation_manager = ConversationManager(
            conversation_timeout=settings.conversation_timeout_seconds,
            max_context_turns=settings.max_context_turns,
            confidence_threshold=settings.stt_confidence_threshold
        )
        print("   ✅ Conversation manager initialized")
        print(f"   📊 Timeout: {settings.conversation_timeout_seconds}s")
        print(f"   📊 Context turns: {settings.max_context_turns}")
        print(f"   📊 Confidence threshold: {settings.stt_confidence_threshold}")
    except Exception as e:
        print(f"   ❌ Conversation manager failed: {e}")
        errors.append(("Conversation Manager", e))
    
    # Test 5: VAD
    print("\n🧪 Test 5: Voice Activity Detection (VAD)")
    try:
        from services.io_service.vad_utils import VADStateManager
        vad_manager = VADStateManager(aggressiveness=settings.vad_aggressiveness)
        print("   ✅ VAD initialized")
        print(f"   📊 Aggressiveness: {settings.vad_aggressiveness}")
    except Exception as e:
        print(f"   ❌ VAD initialization failed: {e}")
        errors.append(("VAD", e))
    
    # Test 6: Hotword Detection
    print("\n🧪 Test 6: Hotword Detection (Porcupine)")
    try:
        from services.io_service.hotword_utils import HotwordDetector
        
        if not settings.picovoice_access_key:
            print("   ⚠️ Warning: PICOVOICE_ACCESS_KEY not set in .env")
            print("   📋 You need to:")
            print("      1. Sign up at https://console.picovoice.ai/")
            print("      2. Get your free access key")
            print("      3. Add to .env: PICOVOICE_ACCESS_KEY=your_key_here")
            errors.append(("Hotword", "Access key not configured"))
        else:
            hotword_detector = HotwordDetector(
                callback=lambda: None,
                sensitivity=settings.hotword_sensitivity,
                enable_debug=False
            )
            print("   ✅ Hotword detector initialized")
            print(f"   📊 Sensitivity: {settings.hotword_sensitivity}")
    except Exception as e:
        print(f"   ❌ Hotword detection failed: {e}")
        errors.append(("Hotword Detection", e))
    
    # Test 7: Audio Stream (without starting)
    print("\n🧪 Test 7: Audio Stream Configuration")
    try:
        from services.io_service.audio_utils import AudioStream, get_audio_info
        
        audio_info = get_audio_info()
        print("   ✅ Audio system ready")
        print(f"   📊 Default input device: {audio_info['default_input_device']}")
        print(f"   📊 Sample rate: {audio_info['config']['sample_rate']} Hz")
        print(f"   📊 Channels: {audio_info['config']['channels']}")
        
    except Exception as e:
        print(f"   ❌ Audio stream check failed: {e}")
        errors.append(("Audio Stream", e))
    
    # Summary
    print("\n" + "="*70)
    if errors:
        print("❌ STARTUP TEST FAILED")
        print("="*70)
        print(f"\n{len(errors)} error(s) found:\n")
        for component, error in errors:
            print(f"   • {component}: {error}")
        print("\n💡 Fix these issues before running the full service.\n")
        return False
    else:
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎉 All components initialized successfully!")
        print("\n📋 Next steps:")
        print("   1. Start Redis: redis-server (if not running)")
        print("   2. Start IO Service: python -m services.io_service.main")
        print("   3. Test voice input: python services/io_service/test_pipeline.py")
        print("\n   OR use the FastAPI server:")
        print("   uvicorn services.io_service.main:app --reload --port 8001")
        print("\n✨ The service will automatically listen for 'ADRIAN'!\n")
        return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_startup())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

