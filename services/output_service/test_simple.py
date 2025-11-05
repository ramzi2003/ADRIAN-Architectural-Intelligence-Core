"""
Simple test script for Output Service TTS.
Run this while Output Service is running.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.redis_client import get_redis_client
from shared.schemas import ResponseText, MessageType
import uuid


async def test_speak():
    """Send a test message to Redis that Output Service will speak."""
    print("🧪 Testing Output Service TTS via Redis...")
    print("   Make sure Output Service is running on port 8006!\n")
    
    redis_client = await get_redis_client()
    
    # Create a test response
    response = ResponseText(
        message_id=str(uuid.uuid4()),
        correlation_id=str(uuid.uuid4()),
        message_type=MessageType.RESPONSE_TEXT,
        text="At once, Sir. This is a test message from Redis.",
        should_speak=True,
        timestamp=datetime.utcnow()
    )
    
    # Publish to Redis
    await redis_client.publish("responses", response.model_dump_json())
    print(f"✅ Sent message: '{response.text}'")
    print("   Listen for ADRIAN's voice...")
    
    await asyncio.sleep(5)  # Wait for TTS
    await redis_client.disconnect()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_speak())

