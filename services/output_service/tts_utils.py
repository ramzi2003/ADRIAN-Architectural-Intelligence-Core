"""
Text-to-Speech utilities for ADRIAN Output Service.
Uses Coqui TTS for high-quality, natural voice synthesis.
"""
import numpy as np
from typing import Optional
from pathlib import Path
import tempfile

from shared.logging_config import get_logger

logger = get_logger("output-service.tts")

# Global TTS instance
_tts = None


def get_tts_engine():
    """
    Get or initialize the Coqui TTS engine.
    Lazy loading to avoid startup delays.
    """
    global _tts
    
    if _tts is None:
        try:
            logger.info("Initializing Coqui TTS engine...")
            from TTS.api import TTS
            
            # Use a pre-trained model
            # tts_models/en/ljspeech/tacotron2-DDC - Fast, good quality
            # tts_models/en/vctk/vits - Multi-speaker, very good quality
            # tts_models/en/jenny/jenny - Female voice, natural
            
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            
            _tts = TTS(model_name=model_name, progress_bar=False)
            logger.info(f"TTS engine initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            logger.warning("TTS will not be available")
            _tts = False  # Mark as failed
    
    return _tts if _tts is not False else None


def text_to_speech(text: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Convert text to speech and save to file.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save audio file (None = temp file)
    
    Returns:
        Path to generated audio file, or None if failed
    """
    tts = get_tts_engine()
    
    if tts is None:
        logger.error("TTS engine not available")
        return None
    
    try:
        # Create output path if not provided
        if output_path is None:
            temp_dir = Path(tempfile.gettempdir()) / "adrian_tts"
            temp_dir.mkdir(exist_ok=True)
            output_path = str(temp_dir / "response.wav")
        
        logger.info(f"Generating speech for: '{text[:50]}...'")
        
        # Generate speech
        tts.tts_to_file(
            text=text,
            file_path=output_path
        )
        
        logger.info(f"Speech generated: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate speech: {e}")
        return None


def text_to_speech_array(text: str, sample_rate: int = 22050) -> Optional[np.ndarray]:
    """
    Convert text to speech and return as numpy array.
    
    Args:
        text: Text to convert
        sample_rate: Desired sample rate
    
    Returns:
        Audio data as numpy array, or None if failed
    """
    tts = get_tts_engine()
    
    if tts is None:
        logger.error("TTS engine not available")
        return None
    
    try:
        logger.info(f"Generating speech array for: '{text[:50]}...'")
        
        # Generate speech to WAV in memory
        import io
        import soundfile as sf
        
        # Create temporary file
        temp_path = text_to_speech(text)
        
        if temp_path:
            # Load and return as array
            audio_data, sr = sf.read(temp_path, dtype='float32')
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            return audio_data
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to generate speech array: {e}")
        return None


def speak(text: str):
    """
    Convenience function to speak text immediately.
    Generates speech and plays it through default speakers.
    
    Args:
        text: Text to speak
    """
    from services.io_service.audio_utils import play_audio, load_audio
    
    logger.info(f"Speaking: '{text[:50]}...'")
    
    # Generate speech file
    audio_file = text_to_speech(text)
    
    if audio_file:
        # Load and play
        audio_data, sample_rate = load_audio(audio_file)
        play_audio(audio_data, sample_rate=sample_rate)
        
        # Clean up temp file
        Path(audio_file).unlink()
    else:
        logger.error("Failed to speak text")


def get_available_voices() -> List[str]:
    """
    Get list of available TTS voices/models.
    """
    tts = get_tts_engine()
    
    if tts is None:
        return []
    
    try:
        return tts.list_models()
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        return []


def test_tts():
    """Test TTS functionality."""
    test_text = "At once, Sir. I am ADRIAN, your AI assistant."
    
    logger.info("Testing TTS...")
    print(f"\nTesting Coqui TTS with text: '{test_text}'")
    
    # Generate and play
    speak(test_text)
    
    print("TTS test complete!")

