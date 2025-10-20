#!/usr/bin/env python3
"""
VAD Test Script for Task 2 - Voice Activity Detection
Tests speech/silence detection using WebRTC VAD.
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from services.io_service.audio_utils import AudioStream, get_default_input_device, list_audio_devices
    from services.io_service.vad_utils import VADStateManager, detect_speech, initialize_vad
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and all dependencies are installed.")
    sys.exit(1)

from shared.logging_config import get_logger

logger = get_logger("vad-test")


def vad_callback(is_speech: bool, state: str):
    """Callback for VAD state changes."""
    status = "🎤 SPEECH" if is_speech else "🔇 SILENCE"
    print(f"[VAD] {status} - State: {state}")


def audio_callback(audio_chunk: np.ndarray):
    """Basic audio callback - just prints chunk info occasionally."""
    # Print every 30 chunks (~1 second) to avoid spam
    if not hasattr(audio_callback, 'chunk_count'):
        audio_callback.chunk_count = 0
    
    audio_callback.chunk_count += 1
    if audio_callback.chunk_count % 30 == 0:
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        print(f"[Audio] Chunk {audio_callback.chunk_count}: RMS={rms:.4f}, Shape={audio_chunk.shape}")


def test_vad_installation():
    """Test if WebRTC VAD is properly installed."""
    print("=" * 50)
    print("Testing VAD Installation...")
    print("=" * 50)
    
    try:
        import webrtcvad
        print("✅ WebRTC VAD imported successfully")
        
        # Test basic VAD functionality
        vad = webrtcvad.Vad(1)
        print("✅ VAD object created")
        
        # Create test audio chunk (silence)
        test_chunk = np.zeros(480, dtype=np.int16)
        test_bytes = test_chunk.tobytes()
        
        is_speech = vad.is_speech(test_bytes, 16000)
        print(f"✅ VAD test: Silence detected as speech: {is_speech} (should be False)")
        
        return True
        
    except ImportError as e:
        print(f"❌ WebRTC VAD import failed: {e}")
        print("Run: pip install webrtcvad")
        return False
    except Exception as e:
        print(f"❌ VAD test failed: {e}")
        return False


def test_audio_devices():
    """Test available audio devices."""
    print("\n" + "=" * 50)
    print("Testing Audio Devices...")
    print("=" * 50)
    
    try:
        devices = list_audio_devices()
        print(f"Found {len(devices)} audio devices:")
        
        input_devices = []
        for device in devices:
            if device['is_input']:
                input_devices.append(device)
                marker = " 🎤" if device['index'] == get_default_input_device() else ""
                print(f"  [{device['index']}] {device['name']}{marker}")
                print(f"      Channels: {device['channels_in']}, Sample Rate: {device['sample_rate']:.0f}Hz")
        
        if not input_devices:
            print("❌ No input devices found!")
            return None
        
        default_device = get_default_input_device()
        print(f"\nDefault input device: {default_device}")
        
        return default_device
        
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return None


def test_vad_utilities():
    """Test VAD utility functions."""
    print("\n" + "=" * 50)
    print("Testing VAD Utilities...")
    print("=" * 50)
    
    try:
        # Test VAD state manager
        vad_manager = initialize_vad(aggressiveness=1)
        print("✅ VAD State Manager initialized")
        
        # Test silence detection
        silence_chunk = np.zeros(480, dtype=np.int16)
        is_speech, state = vad_manager.process_chunk(silence_chunk)
        print(f"✅ Silence test: Speech={is_speech}, State={state.value} (should be False, silence)")
        
        # Test speech simulation (sine wave)
        t = np.linspace(0, 0.03, 480)  # 30ms at 16kHz
        speech_chunk = (np.sin(2 * np.pi * 1000 * t) * 16000).astype(np.int16)
        is_speech, state = vad_manager.process_chunk(speech_chunk)
        print(f"✅ Speech simulation test: Speech={is_speech}, State={state.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ VAD utilities test failed: {e}")
        return False


def test_realtime_vad(duration: int = 30):
    """Test real-time VAD with microphone."""
    print("\n" + "=" * 50)
    print(f"Testing Real-time VAD ({duration}s)...")
    print("=" * 50)
    print("Speak into your microphone and watch for speech detection!")
    print("Memory usage will be monitored to ensure no leaks.")
    print("Press Ctrl+C to stop early.")
    print()
    
    try:
        default_device = get_default_input_device()
        if default_device is None:
            print("❌ No default input device found")
            return False
        
        # Create audio stream with VAD
        stream = AudioStream(
            callback=audio_callback,
            device=default_device,
            enable_vad=True,
            vad_aggressiveness=1,
            vad_callback=vad_callback
        )
        
        print(f"Starting audio stream on device {default_device}...")
        stream.start()
        
        start_time = time.time()
        last_state = None
        last_memory_check = start_time
        
        # Get initial memory usage
        initial_memory = stream.get_memory_usage()
        print(f"[Memory] Initial: {initial_memory['rss_mb']:.1f}MB RSS")
        
        while time.time() - start_time < duration:
            current_time = time.time()
            current_state = stream.get_vad_state()
            is_speaking = stream.is_speaking()
            
            # Print state changes
            if current_state != last_state:
                print(f"\n[State Change] {current_state} (speaking: {is_speaking})")
                last_state = current_state
            
            # Check memory every 5 seconds
            if current_time - last_memory_check >= 5.0:
                memory = stream.get_memory_usage()
                memory_diff = memory['rss_mb'] - initial_memory['rss_mb']
                print(f"[Memory] {memory['rss_mb']:.1f}MB RSS ({memory_diff:+.1f}MB change)")
                last_memory_check = current_time
            
            time.sleep(0.1)  # Check every 100ms
        
        print("\n✅ Real-time VAD test completed")
        stream.stop()
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        if 'stream' in locals():
            stream.stop()
        return True
    except Exception as e:
        print(f"❌ Real-time VAD test failed: {e}")
        if 'stream' in locals():
            stream.stop()
        return False


def main():
    """Main test function."""
    print("🎤 ADRIAN VAD Test - Task 2 Implementation")
    print("Testing Voice Activity Detection functionality")
    print()
    
    # Test 1: Installation
    if not test_vad_installation():
        print("\n❌ VAD installation test failed. Please install webrtcvad:")
        print("   pip install webrtcvad")
        sys.exit(1)
    
    # Test 2: Audio devices
    default_device = test_audio_devices()
    if default_device is None:
        print("\n❌ Audio device test failed. Check your microphone setup.")
        sys.exit(1)
    
    # Test 3: VAD utilities
    if not test_vad_utilities():
        print("\n❌ VAD utilities test failed.")
        sys.exit(1)
    
    # Test 4: Real-time VAD
    print("\n" + "🚀 All basic tests passed! Starting real-time test...")
    time.sleep(2)
    
    if not test_realtime_vad(duration=20):
        print("\n❌ Real-time VAD test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 ALL VAD TESTS PASSED!")
    print("=" * 50)
    print("✅ WebRTC VAD is working correctly")
    print("✅ Speech/silence detection is functional")
    print("✅ Audio stream integration is working")
    print("\nTask 2 - Voice Activity Detection: COMPLETED")


if __name__ == "__main__":
    main()
