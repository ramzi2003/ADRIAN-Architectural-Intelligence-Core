"""
Quick microphone test to check audio levels
"""
import pyaudio
import numpy as np
import time

def test_microphone(duration=10):
    """Test microphone and show real-time audio levels"""
    
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    
    print("🎤 Microphone Test Starting...")
    print("=" * 60)
    print("Speak into your microphone for 10 seconds...")
    print("You should see bars moving when you speak.")
    print("=" * 60)
    
    p = pyaudio.PyAudio()
    
    # Open microphone stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("\n📊 Audio Level Monitor:")
    print("Target: ████████████████████ (RMS > 0.10)")
    print("-" * 60)
    
    start_time = time.time()
    max_rms = 0.0
    
    try:
        while time.time() - start_time < duration:
            # Read audio chunk
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS (same as STT)
            audio_float = audio_array.astype(np.float32) / np.iinfo(np.int16).max
            rms = np.sqrt(np.mean(audio_float ** 2))
            
            # Track maximum
            if rms > max_rms:
                max_rms = rms
            
            # Visual bar
            bar_length = int(rms * 200)  # Scale for display
            bar = "█" * bar_length
            
            # Color coding
            if rms < 0.03:
                status = "❌ TOO QUIET"
                color = ""
            elif rms < 0.10:
                status = "⚠️  LOW"
                color = ""
            else:
                status = "✅ GOOD"
                color = ""
            
            print(f"\rRMS: {rms:.4f} {bar:<40} {status}", end="", flush=True)
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    print("\n" + "=" * 60)
    print(f"📈 Maximum RMS recorded: {max_rms:.4f}")
    print()
    
    if max_rms < 0.03:
        print("❌ MICROPHONE TOO QUIET!")
        print("   → Increase Windows microphone volume to 100%")
        print("   → Enable microphone boost (+20dB or +30dB)")
        print("   → Check if correct microphone is selected")
    elif max_rms < 0.10:
        print("⚠️  MICROPHONE VOLUME LOW")
        print("   → Increase microphone boost")
        print("   → Speak louder or move closer to mic")
    else:
        print("✅ MICROPHONE WORKING PERFECTLY!")
        print("   → STT should work well with this audio level")
    
    print("=" * 60)

if __name__ == "__main__":
    test_microphone(duration=10)

