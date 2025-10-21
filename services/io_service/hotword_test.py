#!/usr/bin/env python3
"""
Test script for hotword detection functionality.
"""
import time
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.io_service.hotword_utils import test_hotword_detection, HotwordDetector, get_adrian_model_info
from services.io_service.audio_utils import AudioStream, get_audio_info
from shared.logging_config import setup_logging

logger = setup_logging("hotword-test")


def test_hotword_detector_class():
    """Test the HotwordDetector class directly."""
    print("\n🔍 Testing HotwordDetector class...")
    
    try:
        # Create detector
        detector = HotwordDetector(
            callback=lambda: print("🎯 Hotword detected!"),
            sensitivity=0.7,
            enable_debug=True
        )
        
        # Test initialization
        status = detector.get_status()
        print(f"✅ Detector created: {status}")
        
        # Test start
        if detector.start():
            print("✅ Hotword detector started successfully")
            
            # Show status
            status = detector.get_status()
            print(f"📊 Status: {status}")
            
            # Stop
            detector.stop()
            print("✅ Hotword detector stopped")
            
        else:
            print("❌ Failed to start hotword detector")
            return False
            
    except Exception as e:
        print(f"❌ Error testing HotwordDetector: {e}")
        return False
    
    return True


def test_audio_stream_integration():
    """Test AudioStream with hotword detection."""
    print("\n🔍 Testing AudioStream integration...")
    
    try:
        # Test audio info first
        audio_info = get_audio_info()
        print(f"📱 Audio devices available: {len([d for d in audio_info['devices'] if d['is_input']])}")
        
        # Create stream with hotword detection
        hotword_count = [0]  # Use list to allow modification in nested function
        
        def on_hotword():
            hotword_count[0] += 1
            print(f"🎯 Hotword detected! (count: {hotword_count[0]})")
        
        def audio_callback(audio_data):
            # Just pass through - hotword detection happens internally
            pass
        
        stream = AudioStream(
            callback=audio_callback,
            enable_vad=True,
            enable_hotword=True,
            hotword_sensitivity=0.7,
            hotword_callback=on_hotword
        )
        
        print("✅ AudioStream with hotword detection created")
        
        # Check if hotword is enabled
        if stream.is_hotword_enabled():
            print("✅ Hotword detection enabled in AudioStream")
            
            # Get status
            hotword_status = stream.get_hotword_status()
            print(f"📊 Hotword status: {hotword_status}")
            
            # Tell user what keyword to say
            keywords = hotword_status.get('keywords', ['unknown'])
            print(f"🗣️  Say one of these keywords: {', '.join(keywords)}")
            
        else:
            print("⚠️  Hotword detection not enabled in AudioStream")
        
        # Test starting the stream
        print("🎤 Starting audio stream...")
        stream.start()
        
        if stream.is_running:
            print("✅ Audio stream started successfully")
            keywords = stream.get_hotword_status().get('keywords', ['computer'])
            print(f"👂 Listening for hotword... Say: {', '.join(keywords)}")
            print("📝 Press Ctrl+C to stop")
            
            try:
                # Run for a short test period
                for i in range(30):  # 30 seconds
                    time.sleep(1)
                    if i % 5 == 0:
                        print(f"⏰ {30-i} seconds remaining...")
                        
            except KeyboardInterrupt:
                print("\n⏹️  Stopping test...")
        
        # Stop stream
        stream.stop()
        print("✅ Audio stream stopped")
        
        print(f"📈 Total hotword detections: {hotword_count[0]}")
        
    except Exception as e:
        print(f"❌ Error testing AudioStream integration: {e}")
        return False
    
    return True


def test_porcupine_availability():
    """Test if Porcupine is properly installed."""
    print("\n🔍 Testing Porcupine availability...")
    
    try:
        test_result = test_hotword_detection()
        if test_result:
            print("✅ Porcupine is working correctly")
            return True
        else:
            print("❌ Porcupine test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Porcupine availability: {e}")
        print("💡 Make sure you have installed Porcupine: pip install pvporcupine")
        return False


def main():
    """Run all hotword detection tests."""
    print("🚀 ADRIAN Hotword Detection Test Suite")
    print("=" * 50)
    
    # Show ADRIAN model information first
    adrian_info = get_adrian_model_info()
    
    if adrian_info["model_exists"]:
        print("\n✅ ADRIAN Model Status:")
        print("🎉 Custom 'ADRIAN' model found! Ready to use.")
        print()
    else:
        print("\n🔍 ADRIAN Model Status:")
        print("⚠️  Custom 'ADRIAN' model not found - creating one now!")
        print("\n📋 Step-by-step instructions to create 'ADRIAN' wake word:")
        for i, instruction in enumerate(adrian_info["instructions"], 1):
            print(f"   {instruction}")
        print(f"\n💾 Model file locations to check:")
        for location in adrian_info["model_locations"][:2]:
            print(f"   📁 {location}")
        print()
    
    results = []
    
    # Test 1: Porcupine availability
    results.append(("Porcupine Availability", test_porcupine_availability()))
    
    # Test 2: HotwordDetector class
    results.append(("HotwordDetector Class", test_hotword_detector_class()))
    
    # Test 3: AudioStream integration  
    results.append(("AudioStream Integration", test_audio_stream_integration()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Hotword detection is ready to use.")
        print("\n💡 To use:")
        print("   1. Start the IO service: python -m services.io_service.main")
        print("   2. POST to /input/voice/start to begin listening")
        print("   3. Say 'ADRIAN' to trigger the system")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("💡 Make sure Porcupine is installed: pip install pvporcupine")


if __name__ == "__main__":
    main()
