"""
Test the cloned JARVIS voice with TTS processor.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.io_service.tts_utils import TTSProcessor, TTSConfig
from shared.logging_config import setup_logging

logger = setup_logging("jarvis-test")

async def test_jarvis_voice():
    """Test cloned JARVIS voice."""
    logger.info("="*70)
    logger.info("Testing Cloned JARVIS Voice")
    logger.info("="*70)
    
    # Create TTS processor with JARVIS voice cloning
    config = TTSConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        speaker_wav="data/jarvis_voice_samples/jarvis_voice.wav",
        use_voice_cloning=True,
        speed=1.15,
        pitch_shift=-2.5,
        language="en"
    )
    
    tts = TTSProcessor(config=config)
    
    # Load model
    logger.info("\n1. Loading XTTS model with voice cloning...")
    if not tts.load_model():
        logger.error("❌ Failed to load TTS model")
        return
    
    logger.info("✅ TTS model loaded successfully")
    
    # Test phrases (JARVIS-style)
    test_phrases = [
        "Good morning, Sir. All systems are online.",
        "I'm ready to assist you with any task.",
        "Shall I run a diagnostic check of the system?",
        "Processing your request now, Sir.",
        "Task completed successfully. Anything else I can help you with?"
    ]
    
    logger.info("\n2. Testing JARVIS voice synthesis...")
    for i, phrase in enumerate(test_phrases, 1):
        logger.info(f"\n   Test {i}/{len(test_phrases)}: '{phrase}'")
        try:
            tts.speak(phrase, blocking=True)
            logger.info("   ✅ Playback successful")
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
    
    logger.info("\n3. Performance Statistics:")
    stats = tts.get_performance_stats()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    tts.shutdown()
    logger.info("\n✅ JARVIS voice test completed!")
    logger.info("="*70)

if __name__ == "__main__":
    asyncio.run(test_jarvis_voice())

