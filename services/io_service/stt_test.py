#!/usr/bin/env python3
"""
Test script for STT functionality.
Tests Whisper integration, conversation management, and personality error handling.
"""
import asyncio
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.io_service.stt_utils import STTProcessor, test_stt_processor
from services.io_service.conversation_manager import ConversationManager, test_conversation_manager
from services.io_service.personality_error_handler import PersonalityErrorHandler, test_personality_handler
from shared.logging_config import setup_logging

logger = setup_logging("stt-test")


async def test_stt_integration():
    """Test complete STT integration."""
    print("🚀 ADRIAN STT Integration Test Suite")
    print("=" * 50)
    
    # Test 1: STT Processor
    print("\n🔍 Testing STT Processor...")
    try:
        test_stt_processor()
        print("✅ STT Processor test completed")
    except Exception as e:
        print(f"❌ STT Processor test failed: {e}")
        return False
    
    # Test 2: Conversation Manager
    print("\n🔍 Testing Conversation Manager...")
    try:
        test_conversation_manager()
        print("✅ Conversation Manager test completed")
    except Exception as e:
        print(f"❌ Conversation Manager test failed: {e}")
        return False
    
    # Test 3: Personality Error Handler
    print("\n🔍 Testing Personality Error Handler...")
    try:
        test_personality_handler()
        print("✅ Personality Error Handler test completed")
    except Exception as e:
        print(f"❌ Personality Error Handler test failed: {e}")
        return False
    
    # Test 4: Integration Test
    print("\n🔍 Testing Integration...")
    try:
        await test_integration_flow()
        print("✅ Integration test completed")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    print("\n🎉 All STT tests passed!")
    return True


async def test_integration_flow():
    """Test the complete integration flow."""
    print("  Testing complete integration flow...")
    
    # Initialize components
    stt_processor = STTProcessor()
    conversation_manager = ConversationManager()
    personality_handler = PersonalityErrorHandler()
    
    # Test conversation start
    conv_id = conversation_manager.start_conversation("test_user")
    print(f"    ✅ Conversation started: {conv_id}")
    
    # Test audio quality analysis
    dummy_audio = np.random.randint(-1000, 1000, size=16000 * 2, dtype=np.int16)  # 2 seconds
    quality, score = stt_processor.analyze_audio_quality(dummy_audio)
    print(f"    ✅ Audio quality analysis: {quality.value} (score: {score:.2f})")
    
    # Test error response generation
    from services.io_service.stt_utils import STTError, AudioQuality
    error_response = personality_handler.get_error_response(
        STTError.LOW_CONFIDENCE, 
        AudioQuality.BACKGROUND_NOISE
    )
    print(f"    ✅ Error response: {error_response}")
    
    # Test conversation end
    conversation_manager.end_conversation()
    print(f"    ✅ Conversation ended")
    
    print("    ✅ Integration flow test completed")


def test_audio_quality_detection():
    """Test audio quality detection with different scenarios."""
    print("\n🔍 Testing Audio Quality Detection...")
    
    processor = STTProcessor()
    
    # Test scenarios
    test_cases = [
        ("Silent audio", np.zeros(16000, dtype=np.int16)),
        ("Very quiet", np.random.randint(-100, 100, size=16000, dtype=np.int16)),
        ("Normal speech", np.random.randint(-1000, 1000, size=16000, dtype=np.int16)),
        ("Loud speech", np.random.randint(-5000, 5000, size=16000, dtype=np.int16)),
        ("Clipped audio", np.random.randint(-32000, 32000, size=16000, dtype=np.int16)),
        ("Too short", np.random.randint(-1000, 1000, size=1000, dtype=np.int16)),
    ]
    
    for name, audio in test_cases:
        quality, score = processor.analyze_audio_quality(audio)
        print(f"    {name}: {quality.value} (score: {score:.2f})")
    
    print("✅ Audio quality detection test completed")


def test_personality_responses():
    """Test personality response generation."""
    print("\n🔍 Testing Personality Responses...")
    
    handler = PersonalityErrorHandler()
    
    # Test different error scenarios
    from services.io_service.stt_utils import STTError, AudioQuality
    
    scenarios = [
        (STTError.LOW_CONFIDENCE, AudioQuality.CLEAR),
        (STTError.BACKGROUND_NOISE, AudioQuality.BACKGROUND_NOISE),
        (STTError.MUSIC_DETECTED, AudioQuality.MUSIC),
        (STTError.MULTIPLE_SPEAKERS, AudioQuality.CLEAR),
        (STTError.TOO_SHORT, AudioQuality.TOO_SHORT),
    ]
    
    for error_type, audio_quality in scenarios:
        response = handler.get_error_response(error_type, audio_quality)
        print(f"    {error_type.value}: {response}")
    
    print("✅ Personality responses test completed")


async def main():
    """Main test function."""
    print("🧪 ADRIAN STT System Test")
    print("=" * 40)
    
    # Test individual components
    test_audio_quality_detection()
    test_personality_responses()
    
    # Test complete integration
    success = await test_stt_integration()
    
    if success:
        print("\n🎉 All tests passed! STT system is ready.")
        print("\n💡 Next steps:")
        print("   1. Install Whisper: pip install openai-whisper")
        print("   2. Test with real audio: python -m services.io_service.main")
        print("   3. Integrate with audio pipeline")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
