"""
Fix corrupted TTS model by deleting and re-downloading.
"""
import os
import shutil
from pathlib import Path

# TTS models are stored in user's home directory
tts_cache = Path.home() / ".local" / "share" / "tts"

print("=" * 60)
print("TTS Model Fix Script")
print("=" * 60)

if tts_cache.exists():
    print(f"\n✅ Found TTS cache: {tts_cache}")
    print(f"   Size: {sum(f.stat().st_size for f in tts_cache.rglob('*') if f.is_file()) / (1024*1024):.1f} MB")
    
    response = input("\nDelete corrupted models and re-download? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\n🗑️  Deleting TTS cache...")
        shutil.rmtree(tts_cache)
        print("✅ TTS cache deleted")
        print("\n📥 Now run the test again to download fresh models:")
        print("   python services/io_service/test_tts.py --test basic")
    else:
        print("\n❌ Cancelled")
else:
    print(f"\n❌ TTS cache not found at: {tts_cache}")
    print("   Models will be downloaded on first use")

print("=" * 60)

