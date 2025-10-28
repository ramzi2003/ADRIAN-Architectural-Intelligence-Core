"""
Test all male British voices from VCTK model.
Helps you choose the best JARVIS-like voice.
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

logger = setup_logging("voice-test")

# Male British speakers in VCTK model
# Source: https://datashare.ed.ac.uk/handle/10283/3443
# Final selected Male British speakers for JARVIS testing
MALE_BRITISH_SPEAKERS = {
    "p230": "English (Stockton-on-tees) - Male",
    "p234": "English (Manchester) - Male"
}

# Best candidates for JARVIS-like voice (deep, authoritative, formal British)
RECOMMENDED_VOICES = ["p230", "p234"]

TEST_PHRASE = (
    "Good evening. Allow me to introduce myself. "
    "I am ADRIAN — your advanced artificial intelligence. "
    "At your discretion, I will manage operations, curate information, and execute instructions with absolute precision. "
    "I remain at your service, twenty-four hours a day, seven days a week. "
    "How would you like to proceed, sir?"
)

async def test_voice(speaker_id: str, description: str, tts: TTSProcessor):
    """Test a single voice."""
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {speaker_id} - {description}")
        logger.info(f"{'='*70}")
        
        # Update speaker
        tts.config.speaker = speaker_id
        
        # Synthesize and play
        audio = tts.synthesize_speech(TEST_PHRASE)
        if audio is not None:
            logger.info(f"🔊 Playing {speaker_id}...")
            tts.play_audio(audio)
            logger.info(f"✅ {speaker_id} complete")
        else:
            logger.error(f"❌ Failed to synthesize {speaker_id}")
            
    except Exception as e:
        logger.error(f"❌ Error testing {speaker_id}: {e}")

async def test_all_voices():
    """Test all male British voices."""
    logger.info("="*70)
    logger.info("VCTK Male British Voice Comparison Tool")
    logger.info("="*70)
    logger.info(f"\nTest phrase: '{TEST_PHRASE}'")
    logger.info(f"\nTesting {len(MALE_BRITISH_SPEAKERS)} male British voices...")
    
    # Create TTS processor with JARVIS effects
    config = TTSConfig(
        model_name="tts_models/en/vctk/vits",
        speaker="p326",  # Will be changed for each test
        use_voice_cloning=False,  # Disable XTTS cloning for VCTK tests
        speaker_wav=None,         # Ensure no reference audio is used
        language=None,            # Do not pass language to VITS
        speed=1.0,                 # Normal pace
        pitch_shift=-2.5          # JARVIS pitch
    )
    
    tts = TTSProcessor(config=config)
    
    # Load model
    logger.info("\nLoading TTS model...")
    if not tts.load_model():
        logger.error("❌ Failed to load TTS model")
        return
    
    logger.info("✅ TTS model loaded\n")
    
    # First test recommended voices
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDED VOICES (Best for JARVIS)")
    logger.info("="*70)
    
    for speaker_id in RECOMMENDED_VOICES:
        if speaker_id in MALE_BRITISH_SPEAKERS:
            await test_voice(speaker_id, MALE_BRITISH_SPEAKERS[speaker_id], tts)
            input("\nPress Enter to continue to next voice...")
    
    # Ask if user wants to test all voices
    logger.info("\n" + "="*70)
    test_all = input("\nTest ALL voices? (y/n): ").lower()
    
    if test_all == 'y':
        logger.info("\n" + "="*70)
        logger.info("ALL VOICES")
        logger.info("="*70)
        
        for speaker_id, description in MALE_BRITISH_SPEAKERS.items():
            if speaker_id not in RECOMMENDED_VOICES:
                await test_voice(speaker_id, description, tts)
                input("\nPress Enter to continue to next voice...")
    
    # Shutdown
    tts.shutdown()
    
    logger.info("\n" + "="*70)
    logger.info("Voice testing complete!")
    logger.info("="*70)
    logger.info("\nTo change voice, update 'tts_speaker' in shared/config.py")
    logger.info("Example: tts_speaker: str = 'p360'")

if __name__ == "__main__":
    asyncio.run(test_all_voices())

