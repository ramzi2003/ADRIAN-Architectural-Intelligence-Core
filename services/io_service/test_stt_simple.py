"""
Simple STT test to verify Whisper is working correctly.
"""
import numpy as np
import asyncio
from services.io_service.stt_utils import STTProcessor

async def test_stt():
    """Test STT with a simple audio sample."""
    print("Initializing STT processor...")
    stt = STTProcessor(model_name="base", device="cpu")
    
    if not stt.load_model():
        print("Failed to load model!")
        return
    
    print("✅ Model loaded successfully")
    
    # Create a test audio signal (1 second of sine wave at 440Hz - A note)
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_float = 0.3 * np.sin(2 * np.pi * frequency * t)  # 30% amplitude
    audio_int16 = (audio_float * np.iinfo(np.int16).max).astype(np.int16)
    
    print(f"\nTest audio: {len(audio_int16)} samples ({duration}s)")
    print(f"RMS energy: {np.sqrt(np.mean((audio_int16.astype(np.float32) / np.iinfo(np.int16).max) ** 2)):.4f}")
    
    print("\nTranscribing...")
    transcription, confidence, quality, error = await stt.transcribe_audio(audio_int16)
    
    print(f"\n📊 Results:")
    print(f"  Transcription: '{transcription}'")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Quality: {quality.value}")
    print(f"  Error: {error.value if error else 'None'}")
    
    if not transcription:
        print("\n⚠️ No transcription - this is expected for a sine wave (no speech)")
        print("   Whisper correctly detected there's no speech content")
    
    print("\n✅ STT test complete!")

if __name__ == "__main__":
    asyncio.run(test_stt())

