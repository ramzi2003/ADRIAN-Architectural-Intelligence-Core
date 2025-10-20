"""
Audio Infrastructure Test Script for ADRIAN.
Tests microphone input, speaker output, and audio processing.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.io_service.audio_utils import (
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
    record_audio,
    play_audio,
    save_audio,
    load_audio,
    quick_record,
    quick_play,
    get_audio_info,
    SAMPLE_RATE
)

def print_separator(title=""):
    """Print a section separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)
    else:
        print("=" * 70)


def test_audio_devices():
    """Test 1: List all audio devices."""
    print_separator("Test 1: Audio Devices")
    
    devices = list_audio_devices()
    
    print(f"\nFound {len(devices)} audio devices:\n")
    
    for device in devices:
        device_type = []
        if device['is_input']:
            device_type.append("INPUT")
        if device['is_output']:
            device_type.append("OUTPUT")
        
        type_str = " + ".join(device_type) if device_type else "N/A"
        
        print(f"  [{device['index']}] {device['name']}")
        print(f"      Type: {type_str}")
        print(f"      Channels In: {device['channels_in']}, Out: {device['channels_out']}")
        print(f"      Sample Rate: {device['sample_rate']} Hz")
        print()
    
    # Show defaults
    default_in = get_default_input_device()
    default_out = get_default_output_device()
    
    print(f"Default Input Device: {default_in}")
    print(f"Default Output Device: {default_out}")
    
    if default_in is not None:
        input_device = devices[default_in]
        print(f"  -> {input_device['name']}")
    
    if default_out is not None:
        output_device = devices[default_out]
        print(f"  -> {output_device['name']}")


def test_microphone_recording():
    """Test 2: Record audio from microphone."""
    print_separator("Test 2: Microphone Recording")
    
    print("\nPreparing to record from default microphone...")
    print("SPEAK NOW! Say something like: 'Hello ADRIAN, this is a test.'")
    print("Recording for 5 seconds...\n")
    
    # Use default microphone (built-in laptop mic)
    filename = quick_record(seconds=5, filename="test_audio/test_recording.wav")
    
    print(f"\nRecording saved to: {filename}")
    print(f"File size: {Path(filename).stat().st_size} bytes")


def test_audio_playback():
    """Test 3: Play recorded audio."""
    print_separator("Test 3: Audio Playback")
    
    filename = "test_audio/test_recording.wav"
    
    if not Path(filename).exists():
        print(f"\nERROR: Recording file not found: {filename}")
        print("Run Test 2 first to create a recording.")
        return
    
    print(f"\nPlaying back your recording from: {filename}")
    print("You should hear what you just said...\n")
    
    quick_play(filename)
    
    print("\nPlayback complete!")


def test_audio_file_operations():
    """Test 4: Load and save audio files."""
    print_separator("Test 4: Audio File Operations")
    
    source_file = "test_audio/test_recording.wav"
    
    if not Path(source_file).exists():
        print("\nSkipping - no test recording available")
        return
    
    print(f"\nLoading audio from: {source_file}")
    
    # Load
    audio_data, sample_rate = load_audio(source_file)
    
    print(f"  Loaded: {len(audio_data)} samples at {sample_rate} Hz")
    print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"  Data type: {audio_data.dtype}")
    print(f"  Shape: {audio_data.shape}")
    
    # Save to new file
    new_file = "test_audio/test_copy.wav"
    save_audio(audio_data, new_file, sample_rate=sample_rate)
    
    print(f"\nCopied to: {new_file}")
    print(f"  Size: {Path(new_file).stat().st_size} bytes")


def test_audio_info():
    """Test 5: Display audio system information."""
    print_separator("Test 5: Audio System Info")
    
    info = get_audio_info()
    
    print("\nAudio Configuration:")
    print(f"  Sample Rate: {info['config']['sample_rate']} Hz")
    print(f"  Channels: {info['config']['channels']} (mono)")
    print(f"  Data Type: {info['config']['dtype']}")
    print(f"  Chunk Size: {info['config']['chunk_size']} samples")
    
    print("\nDefault Devices:")
    print(f"  Input: {info['default_input_device']}")
    print(f"  Output: {info['default_output_device']}")


def run_all_tests():
    """Run all audio tests."""
    print("\n")
    print("*" * 70)
    print("  ADRIAN Audio Infrastructure Test Suite")
    print("*" * 70)
    
    # Create test directory
    Path("test_audio").mkdir(exist_ok=True)
    
    try:
        # Run tests
        test_audio_devices()
        
        input("\nPress Enter to start microphone recording test...")
        test_microphone_recording()
        
        input("\nPress Enter to play back your recording...")
        test_audio_playback()
        
        test_audio_file_operations()
        test_audio_info()
        
        print_separator("All Tests Complete")
        print("\nSUCCESS: Audio infrastructure is working!")
        print("\nNext steps:")
        print("  1. Audio input/output verified")
        print("  2. Ready for Task 2: Voice Activity Detection")
        print("  3. Then Task 3: Hotword Detection")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check microphone is connected and not muted")
        print("  2. Check speakers are connected and volume is up")
        print("  3. Try selecting a different audio device")
        return False
    
    return True


if __name__ == "__main__":
    print("\nADRIAN Audio Test")
    print("Make sure your microphone and speakers are ready!\n")
    
    success = run_all_tests()
    
    sys.exit(0 if success else 1)

