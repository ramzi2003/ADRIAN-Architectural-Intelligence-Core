"""
Clone JARVIS voice using Coqui TTS voice cloning.
Requires: JARVIS audio samples in data/jarvis_voice_samples/
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from TTS.api import TTS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-clone")

def clone_jarvis_voice():
    """Clone JARVIS voice from audio samples."""
    
    logger.info("="*70)
    logger.info("JARVIS Voice Cloning Tool")
    logger.info("="*70)
    
    # Path to voice samples
    samples_dir = Path("data/jarvis_voice_samples")
    
    if not samples_dir.exists():
        logger.error(f"❌ Directory not found: {samples_dir}")
        logger.info("\nCreate directory and add JARVIS audio samples:")
        logger.info(f"  mkdir -p {samples_dir}")
        logger.info(f"  # Copy JARVIS WAV files to {samples_dir}/")
        return
    
    # Get all WAV files
    audio_files = list(samples_dir.glob("*.wav"))
    
    if not audio_files:
        logger.error(f"❌ No WAV files found in {samples_dir}")
        logger.info("\nAdd JARVIS audio samples (WAV format) to:")
        logger.info(f"  {samples_dir}/")
        return
    
    logger.info(f"\n✅ Found {len(audio_files)} audio samples:")
    for i, file in enumerate(audio_files, 1):
        logger.info(f"  {i}. {file.name}")
    
    # Load TTS model with voice cloning support
    logger.info("\n📥 Loading TTS model with voice cloning...")
    logger.info("   (This may take a few minutes on first run)")
    
    try:
        # Use XTTS model for voice cloning
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        logger.info("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.info("\nTrying alternative voice cloning model...")
        try:
            # Alternative: YourTTS (faster but less quality)
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)
            logger.info("✅ Alternative model loaded")
        except Exception as e2:
            logger.error(f"❌ Alternative model also failed: {e2}")
            return
    
    # Test voice cloning
    logger.info("\n🎙️ Testing voice cloning...")
    
    test_text = "Good morning, Sir. All systems are online and ready for your command."
    
    # Use first audio file as speaker reference
    speaker_wav = str(audio_files[0])
    output_file = "data/jarvis_cloned_test.wav"
    
    logger.info(f"\n   Speaker reference: {audio_files[0].name}")
    logger.info(f"   Test text: '{test_text}'")
    logger.info(f"   Output: {output_file}")
    
    try:
        # Clone voice and synthesize
        tts.tts_to_file(
            text=test_text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=output_file
        )
        
        logger.info(f"\n✅ Voice cloning successful!")
        logger.info(f"   Audio saved to: {output_file}")
        logger.info(f"\n🔊 Play the audio to check quality:")
        logger.info(f"   powershell.exe -Command \"(New-Object Media.SoundPlayer '{output_file}').PlaySync()\"")
        
        # Play audio automatically
        import subprocess
        win_path = str(Path(output_file).absolute()).replace("/mnt/c/", "C:\\").replace("/", "\\")
        subprocess.run(
            ["powershell.exe", "-Command", f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"],
            check=False
        )
        
        logger.info("\n" + "="*70)
        logger.info("Next Steps:")
        logger.info("="*70)
        logger.info("1. If quality is good, integrate into TTS processor")
        logger.info("2. If not, add more/better JARVIS audio samples")
        logger.info("3. Try different reference audio files")
        logger.info("\nNote: Voice cloning requires good quality, clean audio samples!")
        
    except Exception as e:
        logger.error(f"❌ Voice cloning failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("- Ensure audio is WAV format, 22050Hz")
        logger.info("- Remove background music/noise")
        logger.info("- Use at least 3-5 seconds of clear speech")

if __name__ == "__main__":
    clone_jarvis_voice()

