"""
Shared Redis client for pub/sub and caching across ADRIAN services.
"""
import redis.asyncio as redis
import json
import logging
from typing import Optional, Callable, Any
from shared.config import get_settings
from shared.schemas import BaseMessage

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client wrapper for ADRIAN services."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
    async def connect(self):
        """Establish connection to Redis."""
        try:
            self.redis = await redis.from_url(
                f"redis://{self.settings.redis_host}:{self.settings.redis_port}/{self.settings.redis_db}",
                password=self.settings.redis_password,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
        logger.info("Disconnected from Redis")
    
    async def publish(self, channel: str, message: BaseMessage):
        """Publish a message to a channel."""
        try:
            message_json = message.model_dump_json()
            await self.redis.publish(channel, message_json)
            logger.debug(f"Published to {channel}: {message.message_type}")
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            raise
    
    async def subscribe(self, channel: str, handler: Callable[[dict], Any]):
        """Subscribe to a channel and process messages with handler."""
        try:
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await handler(data)
                    except Exception as e:
                        logger.error(f"Error processing message from {channel}: {e}")
        except Exception as e:
            logger.error(f"Error in subscription to {channel}: {e}")
            raise
    
    async def set(self, key: str, value: str, expire: Optional[int] = None):
        """Set a key-value pair in Redis."""
        await self.redis.set(key, value, ex=expire)
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value from Redis."""
        return await self.redis.get(key)
    
    async def delete(self, key: str):
        """Delete a key from Redis."""
        await self.redis.delete(key)


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Get or create Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    return _redis_client

