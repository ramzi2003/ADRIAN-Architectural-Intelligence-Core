"""
Security Service - Authentication and Authorization Layer
Handles user authentication, permissions, and secrets management.
"""
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import SecurityCheck, SecurityResult, HealthResponse
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Setup logging
logger = setup_logging("security-service")
settings = get_settings()


# Request models
class AuthenticateRequest(BaseModel):
    """Request to authenticate a user."""
    user_id: str
    auth_method: str = "voice"  # voice, password, pin
    auth_data: Optional[str] = None


class PermissionCheckRequest(BaseModel):
    """Request to check permissions for an action."""
    user_id: str
    action_intent: str


# Background task flag
_listening = False


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _listening
    logger.info("Starting Security Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    
    # Start background listener for security checks
    _listening = True
    asyncio.create_task(listen_for_security_checks())
    
    # TODO: Initialize voiceprint database
    # TODO: Initialize secrets vault
    
    yield
    
    # Shutdown
    logger.info("Shutting down Security Service...")
    _listening = False
    await redis_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="ADRIAN Security Service",
    description="Authentication and Authorization Layer",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        service_name="security-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "voiceprint_db": "stub",
            "secrets_vault": "stub"
        }
    )


@app.post("/auth/authenticate")
async def authenticate(request: AuthenticateRequest):
    """
    Authenticate a user via voice, password, or PIN.
    """
    try:
        result = await verify_authentication(
            request.user_id,
            request.auth_method,
            request.auth_data
        )
        return result
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/check-permission")
async def check_permission(request: PermissionCheckRequest):
    """
    Check if a user has permission to perform an action.
    """
    try:
        approved = await verify_permission(request.user_id, request.action_intent)
        return {
            "user_id": request.user_id,
            "action": request.action_intent,
            "approved": approved
        }
    except Exception as e:
        logger.error(f"Permission check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def listen_for_security_checks():
    """
    Background task that listens to security_check events from Redis.
    """
    logger.info("Started listening for security checks...")
    redis_client = await get_redis_client()
    
    async def handle_security_check(data: dict):
        """Process incoming security check."""
        try:
            check = SecurityCheck(**data)
            logger.info(f"Security check for action: {check.action_intent}")
            
            # Verify permission
            approved = await verify_permission(
                check.user_id or "default_user",
                check.action_intent
            )
            
            # Emit result
            result = SecurityResult(
                message_id=str(uuid.uuid4()),
                correlation_id=check.correlation_id,
                check_id=check.message_id,
                approved=approved,
                reason="Stub approval" if approved else "Stub denial"
            )
            await redis_client.publish("security_results", result)
            
        except Exception as e:
            logger.error(f"Error handling security check: {e}")
    
    # Subscribe and listen
    await redis_client.subscribe("security_checks", handle_security_check)


async def verify_authentication(user_id: str, method: str, auth_data: Optional[str]) -> dict:
    """
    Verify user authentication.
    TODO: Implement actual authentication (voiceprint, password hash, etc.)
    """
    logger.info(f"[STUB] Authenticating user '{user_id}' via {method}")
    
    # Stub: always approve for now
    return {
        "authenticated": True,
        "user_id": user_id,
        "method": method
    }


async def verify_permission(user_id: str, action_intent: str) -> bool:
    """
    Check if user has permission for an action.
    TODO: Implement role-based access control (RBAC).
    """
    logger.info(f"[STUB] Checking permission for '{user_id}' to perform '{action_intent}'")
    
    # Stub: Deny dangerous actions by default
    dangerous_actions = ["delete_file", "shutdown", "install_app", "format_drive"]
    
    if action_intent in dangerous_actions:
        logger.warning(f"Dangerous action '{action_intent}' requires explicit approval")
        return False  # Would require additional auth
    
    return True


# TODO: Voiceprint enrollment and verification
async def enroll_voiceprint(user_id: str, audio_data: bytes):
    """TODO: Enroll user voiceprint for speaker identification."""
    logger.info(f"[STUB] Enrolling voiceprint for user: {user_id}")
    pass


async def verify_voiceprint(audio_data: bytes) -> Optional[str]:
    """TODO: Verify speaker identity from audio and return user_id."""
    logger.info("[STUB] Verifying voiceprint...")
    return "default_user"


# TODO: Secrets vault operations
async def store_secret(key: str, value: str, user_id: str):
    """TODO: Store encrypted secret in vault."""
    logger.info(f"[STUB] Storing secret: {key}")
    pass


async def retrieve_secret(key: str, user_id: str) -> Optional[str]:
    """TODO: Retrieve and decrypt secret from vault."""
    logger.info(f"[STUB] Retrieving secret: {key}")
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.security_service_port,
        log_level=settings.log_level.lower()
    )

