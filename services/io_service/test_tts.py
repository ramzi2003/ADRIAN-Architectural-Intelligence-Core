"""
Test script for TTS functionality.
Tests voice synthesis and playback.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.io_service.tts_utils import TTSProcessor, TTSConfig
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("tts-test")


def test_tts_basic():
    """Test basic TTS functionality."""
    logger.info("=" * 60)
    logger.info("Testing Coqui TTS - Basic Functionality")
    logger.info("=" * 60)
    
    # Create TTS processor with selected voice (p234)
    config = TTSConfig(
        model_name="tts_models/en/vctk/vits",
        speaker="p234",  # Selected voice
        use_voice_cloning=False,  # Disable XTTS cloning
        speaker_wav=None,         # No reference audio
        language=None,            # No language parameter for VITS
        speed=1.0,               # Normal speed
        pitch_shift=-2.5         # JARVIS pitch
    )
    
    tts = TTSProcessor(config=config)
    
    # Load model
    logger.info("\n1. Loading TTS model...")
    if not tts.load_model():
        logger.error("❌ Failed to load TTS model")
        return False
    
    logger.info("✅ TTS model loaded successfully")
    
    # Test phrases (JARVIS-like)
    test_phrases = [
        "Good morning, Sir. All systems are online.",
        "I'm ready to assist you.",
        "Shall I run a diagnostic check?",
        "Processing your request now.",
        "Task completed successfully, Sir."
    ]
    
    logger.info("\n2. Testing voice synthesis...")
    for i, phrase in enumerate(test_phrases, 1):
        logger.info(f"\n   Test {i}/{len(test_phrases)}: '{phrase}'")
        
        # Synthesize speech
        audio = tts.synthesize_speech(phrase)
        
        if audio is None:
            logger.error(f"   ❌ Failed to synthesize: '{phrase}'")
            continue
        
        logger.info(f"   ✅ Synthesized {len(audio)} samples")
        
        # Play audio
        logger.info("   🔊 Playing audio...")
        if tts.play_audio(audio):
            logger.info("   ✅ Playback successful")
        else:
            logger.error("   ❌ Playback failed")
    
    # Show statistics
    logger.info("\n3. Performance Statistics:")
    stats = tts.get_performance_stats()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    # Cleanup
    tts.shutdown()
    logger.info("\n✅ TTS test completed successfully")
    logger.info("=" * 60)
    
    return True


def test_tts_queue():
    """Test TTS playback queue."""
    logger.info("=" * 60)
    logger.info("Testing TTS Playback Queue")
    logger.info("=" * 60)
    
    config = TTSConfig(speed=1.2)
    tts = TTSProcessor(config=config)
    
    if not tts.load_model():
        logger.error("❌ Failed to load TTS model")
        return False
    
    # Queue multiple messages
    messages = [
        "Message one",
        "Message two",
        "Message three"
    ]
    
    logger.info("\nQueuing multiple messages...")
    for msg in messages:
        tts.speak(msg, blocking=False)
        logger.info(f"   Queued: '{msg}'")
    
    # Wait for queue to finish
    logger.info("\nWaiting for playback to complete...")
    tts.playback_queue.join()
    
    logger.info("✅ All messages played")
    
    tts.shutdown()
    logger.info("=" * 60)
    
    return True


def test_voice_models():
    """List available voice models."""
    logger.info("=" * 60)
    logger.info("Available TTS Models")
    logger.info("=" * 60)
    
    tts = TTSProcessor()
    models = tts.get_available_models()
    
    # Filter for English models
    en_models = [m for m in models if '/en/' in m]
    
    logger.info(f"\nFound {len(en_models)} English models:")
    for i, model in enumerate(en_models[:10], 1):  # Show first 10
        logger.info(f"   {i}. {model}")
    
    if len(en_models) > 10:
        logger.info(f"   ... and {len(en_models) - 10} more")
    
    logger.info("\nRecommended models for JARVIS-like voice:")
    recommended = [
        "tts_models/en/ljspeech/tacotron2-DDC",  # Fast, good quality
        "tts_models/en/ljspeech/glow-tts",  # Fast, lighter
        "tts_models/en/vctk/vits",  # Multi-speaker
    ]
    for model in recommended:
        logger.info(f"   • {model}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TTS functionality")
    parser.add_argument(
        "--test",
        choices=["basic", "queue", "models", "all"],
        default="basic",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    try:
        if args.test == "basic":
            test_tts_basic()
        elif args.test == "queue":
            test_tts_queue()
        elif args.test == "models":
            test_voice_models()
        elif args.test == "all":
            test_voice_models()
            print("\n")
            test_tts_basic()
            print("\n")
            test_tts_queue()
    
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)

