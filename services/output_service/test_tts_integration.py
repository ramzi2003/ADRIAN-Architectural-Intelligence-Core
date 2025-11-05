"""
Test script for Output Service TTS integration.
Tests TTS functionality via API endpoints and Redis messages.
"""
import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.redis_client import get_redis_client
from shared.schemas import ResponseText, MessageType
import uuid


async def test_redis_message():
    """Test TTS via Redis message."""
    print("\n=== Testing TTS via Redis Message ===")
    
    redis_client = await get_redis_client()
    
    # Create a ResponseText message
    response = ResponseText(
        message_id=str(uuid.uuid4()),
        correlation_id=str(uuid.uuid4()),
        message_type=MessageType.RESPONSE_TEXT,
        text="At once, Sir. This is a test of the Output Service TTS integration.",
        should_speak=True,
        timestamp=datetime.utcnow()
    )
    
    # Publish to Redis
    await redis_client.publish("responses", response.model_dump_json())
    print(f"✅ Published message to Redis: '{response.text}'")
    print("   Listen for audio playback...")
    
    await asyncio.sleep(5)  # Wait for TTS to process
    
    await redis_client.disconnect()


async def test_multiple_messages():
    """Test queuing multiple messages."""
    print("\n=== Testing TTS Queue with Multiple Messages ===")
    
    redis_client = await get_redis_client()
    
    messages = [
        "First message in the queue.",
        "Second message should play after the first.",
        "Third message completes the queue."
    ]
    
    for i, text in enumerate(messages, 1):
        response = ResponseText(
            message_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE_TEXT,
            text=text,
            should_speak=True,
            timestamp=datetime.utcnow()
        )
        
        await redis_client.publish("responses", response.model_dump_json())
        print(f"✅ Published message {i}/{len(messages)}: '{text}'")
        await asyncio.sleep(0.5)  # Small delay between messages
    
    print("   All messages queued. Listen for sequential audio playback...")
    await asyncio.sleep(10)  # Wait for all TTS to complete
    
    await redis_client.disconnect()


async def main():
    """Run all tests."""
    print("🧪 Output Service TTS Integration Test")
    print("=" * 50)
    
    print("\n⚠️  Make sure Output Service is running on port 8006!")
    print("   Press Enter to start tests...")
    input()
    
    try:
        # Test 1: Single message
        await test_redis_message()
        
        # Wait a bit
        await asyncio.sleep(3)
        
        # Test 2: Multiple messages (queue test)
        await test_multiple_messages()
        
        print("\n✅ All tests completed!")
        print("\n📝 Check the Output Service logs for any errors.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

