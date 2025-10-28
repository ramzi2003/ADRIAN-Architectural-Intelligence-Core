"""
Prepare JARVIS audio file for voice cloning.
Converts MP3 to WAV and checks quality.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-prep")

def prepare_jarvis_audio():
    """Prepare JARVIS audio for voice cloning."""
    
    logger.info("="*70)
    logger.info("JARVIS Audio Preparation Tool")
    logger.info("="*70)
    
    # Get input file path
    logger.info("\nPlace your JARVIS MP3 file in: data/jarvis_voice_samples/")
    logger.info("Example: data/jarvis_voice_samples/jarvis.mp3")
    
    input_path = input("\nEnter path to JARVIS MP3 file: ").strip()
    
    if not Path(input_path).exists():
        logger.error(f"❌ File not found: {input_path}")
        return
    
    logger.info(f"✅ Found file: {input_path}")
    
    # Create output directory
    output_dir = Path("data/jarvis_voice_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "jarvis_voice.wav"
    
    logger.info(f"\n🔄 Converting MP3 to WAV (22050Hz, mono)...")
    logger.info(f"   Output: {output_path}")
    
    # Convert using ffmpeg via WSL/Windows
    try:
        # Try ffmpeg (needs to be installed)
        win_input = str(Path(input_path).absolute()).replace("/mnt/c/", "C:\\").replace("/", "\\")
        win_output = str(output_path.absolute()).replace("/mnt/c/", "C:\\").replace("/", "\\")
        
        # Use PowerShell to run ffmpeg
        cmd = f"ffmpeg -i '{win_input}' -ar 22050 -ac 1 -y '{win_output}'"
        
        result = subprocess.run(
            ["powershell.exe", "-Command", cmd],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and output_path.exists():
            logger.info("✅ Conversion successful!")
            
            # Get file info
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"   File size: {size_mb:.2f} MB")
            
            logger.info("\n" + "="*70)
            logger.info("Next Steps:")
            logger.info("="*70)
            logger.info(f"1. Audio saved to: {output_path}")
            logger.info("2. Run voice cloning:")
            logger.info("   python services/io_service/clone_jarvis_voice.py")
            logger.info("\n⚠️  Note: If audio has background music, cloning quality will be poor")
            logger.info("   Best results = clean voice only, no music/effects")
            
        else:
            raise Exception("Conversion failed")
            
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        logger.info("\n📦 FFmpeg not found. Install it:")
        logger.info("   Windows: Download from https://ffmpeg.org/download.html")
        logger.info("   Or use: winget install ffmpeg")
        logger.info("\n   Alternative: Use online MP3->WAV converter")
        logger.info("   - Upload your MP3")
        logger.info("   - Convert to WAV (22050Hz, mono)")
        logger.info(f"   - Save to: {output_path}")

if __name__ == "__main__":
    prepare_jarvis_audio()

