"""
End-to-end test for voice pipeline: STT → TTS
Tests the complete flow from speech input to speech output.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.redis_client import get_redis_client
from shared.schemas import ResponseText
import uuid
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("e2e-test")


async def test_tts_response():
    """
    Test TTS by publishing a ResponseText event to Redis.
    The IO Service should pick it up and speak it.
    """
    logger.info("=" * 60)
    logger.info("End-to-End Test: TTS Response")
    logger.info("=" * 60)
    
    logger.info("\n📋 Test Setup:")
    logger.info("   1. Make sure IO Service is running")
    logger.info("   2. Make sure Redis is running")
    logger.info("   3. Make sure speakers are on")
    logger.info("")
    
    input("Press Enter when ready to test TTS...")
    
    try:
        # Connect to Redis
        logger.info("\n1. Connecting to Redis...")
        redis = await get_redis_client()
        logger.info("   ✅ Connected to Redis")
        
        # Test phrases (JARVIS-like)
        test_phrases = [
            "Good morning, Sir. All systems are online.",
            "I'm ready to assist you with your tasks.",
            "Shall I run a diagnostic check?",
            "Processing your request now, Sir.",
            "Task completed successfully."
        ]
        
        logger.info(f"\n2. Publishing {len(test_phrases)} test responses...")
        
        for i, phrase in enumerate(test_phrases, 1):
            logger.info(f"\n   Test {i}/{len(test_phrases)}: '{phrase}'")
            
            # Create response event
            response = ResponseText(
                message_id=str(uuid.uuid4()),
                correlation_id=str(uuid.uuid4()),
                text=phrase,
                should_speak=True,
                emotion="confident"
            )
            
            # Publish to Redis
            await redis.publish("responses", response)
            logger.info("   📤 Published to Redis")
            logger.info("   🔊 ADRIAN should be speaking now...")
            
            # Wait for speech to complete (estimate)
            await asyncio.sleep(5)
        
        logger.info("\n✅ All test responses published")
        logger.info("\n" + "=" * 60)
        logger.info("Test Complete!")
        logger.info("=" * 60)
        
        # Disconnect
        await redis.disconnect()
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)


async def test_full_conversation():
    """
    Simulate a full conversation flow.
    """
    logger.info("=" * 60)
    logger.info("End-to-End Test: Full Conversation")
    logger.info("=" * 60)
    
    logger.info("\n📋 Test Scenario:")
    logger.info("   User: 'ADRIAN, what's the weather?'")
    logger.info("   ADRIAN: (processes and responds)")
    logger.info("")
    
    input("Press Enter when ready to test full conversation...")
    
    try:
        redis = await get_redis_client()
        
        # Simulate ADRIAN's response
        responses = [
            "I'm checking the weather for you, Sir.",
            "The current temperature is 72 degrees Fahrenheit.",
            "It's partly cloudy with a chance of rain later today.",
            "Would you like me to check the forecast for the week?"
        ]
        
        logger.info("\n🤖 ADRIAN is responding...")
        
        for response_text in responses:
            logger.info(f"\n   💬 '{response_text}'")
            
            response = ResponseText(
                message_id=str(uuid.uuid4()),
                correlation_id=str(uuid.uuid4()),
                text=response_text,
                should_speak=True,
                emotion="helpful"
            )
            
            await redis.publish("responses", response)
            await asyncio.sleep(4)  # Wait between responses
        
        logger.info("\n✅ Conversation test complete")
        logger.info("=" * 60)
        
        await redis.disconnect()
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)


async def test_error_response():
    """
    Test personality-driven error responses.
    """
    logger.info("=" * 60)
    logger.info("End-to-End Test: Error Response")
    logger.info("=" * 60)
    
    input("Press Enter to test error response...")
    
    try:
        redis = await get_redis_client()
        
        error_responses = [
            "I beg your pardon, Sir. I didn't quite catch that.",
            "My apologies, but there seems to be some background noise interfering.",
            "Could you repeat that, Sir? The audio was a bit unclear."
        ]
        
        logger.info("\n🤖 Testing error responses...")
        
        for error_text in error_responses:
            logger.info(f"\n   💬 '{error_text}'")
            
            response = ResponseText(
                message_id=str(uuid.uuid4()),
                correlation_id=str(uuid.uuid4()),
                text=error_text,
                should_speak=True,
                emotion="apologetic"
            )
            
            await redis.publish("responses", response)
            await asyncio.sleep(4)
        
        logger.info("\n✅ Error response test complete")
        logger.info("=" * 60)
        
        await redis.disconnect()
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)


async def main():
    """Main test menu."""
    print("\n" + "=" * 60)
    print("ADRIAN End-to-End Testing")
    print("=" * 60)
    print("\nAvailable Tests:")
    print("  1. TTS Response (basic)")
    print("  2. Full Conversation")
    print("  3. Error Responses")
    print("  4. All Tests")
    print("  5. Exit")
    print("=" * 60)
    
    choice = input("\nSelect test (1-5): ").strip()
    
    try:
        if choice == "1":
            await test_tts_response()
        elif choice == "2":
            await test_full_conversation()
        elif choice == "3":
            await test_error_response()
        elif choice == "4":
            await test_tts_response()
            print("\n")
            await test_full_conversation()
            print("\n")
            await test_error_response()
        elif choice == "5":
            print("\n👋 Goodbye!")
            return
        else:
            print("\n❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

