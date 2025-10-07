"""
Execution Service - System Control and Integration Layer
Handles OS control, application management, file operations, and integrations.
"""
import asyncio
import uuid
import subprocess
import platform
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import ActionSpec, ActionResult, HealthResponse
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

# Setup logging
logger = setup_logging("execution-service")
settings = get_settings()


# Request models
class ExecuteActionRequest(BaseModel):
    """Request model for direct action execution."""
    intent: str
    parameters: dict = {}


# Background task flag
_listening = False


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _listening
    logger.info("Starting Execution Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")
    
    # Start background listener for action_specs
    _listening = True
    asyncio.create_task(listen_for_actions())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Execution Service...")
    _listening = False
    await redis_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="ADRIAN Execution Service",
    description="System Control and Integration Layer",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        service_name="execution-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "os": platform.system()
        }
    )


@app.post("/execute")
async def execute_action(request: ExecuteActionRequest):
    """
    Synchronous endpoint to execute an action directly.
    Useful for testing without going through Redis.
    """
    try:
        result = await perform_action(request.intent, request.parameters)
        return result
    except Exception as e:
        logger.error(f"Error executing action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def listen_for_actions():
    """
    Background task that listens to action_spec events from Redis.
    """
    logger.info("Started listening for action specs...")
    redis_client = await get_redis_client()
    
    async def handle_action(data: dict):
        """Process incoming action spec."""
        try:
            action_spec = ActionSpec(**data)
            logger.info(f"Received action: {action_spec.intent}")
            
            # Check if permission required (would call security-service)
            if action_spec.requires_permission:
                logger.info(f"Action {action_spec.intent} requires permission check")
                # TODO: Call security-service for permission
                # For now, assume approved
            
            # Execute the action
            result = await perform_action(
                action_spec.intent, 
                action_spec.parameters
            )
            
            # Emit action result
            action_result = ActionResult(
                message_id=str(uuid.uuid4()),
                correlation_id=action_spec.correlation_id,
                action_id=action_spec.message_id,
                success=result["success"],
                result_data=result.get("data"),
                error_message=result.get("error")
            )
            await redis_client.publish("action_results", action_result)
            
        except Exception as e:
            logger.error(f"Error handling action: {e}")
    
    # Subscribe and listen
    await redis_client.subscribe("action_specs", handle_action)


async def perform_action(intent: str, parameters: dict) -> dict:
    """
    Main action dispatcher. Routes to appropriate handler based on intent.
    """
    logger.info(f"Performing action: {intent} with params: {parameters}")
    
    handlers = {
        "open_app": handle_open_app,
        "close_app": handle_close_app,
        "search": handle_search,
        "file_operation": handle_file_operation,
        "system_control": handle_system_control,
        "reminder": handle_reminder,
    }
    
    handler = handlers.get(intent, handle_unknown)
    
    try:
        result = await handler(parameters)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Action failed: {e}")
        return {"success": False, "error": str(e)}


# Action handlers

async def handle_open_app(params: dict) -> dict:
    """
    TODO: Open an application.
    Platform-specific implementation needed.
    """
    app_name = params.get("app_name", "").strip()
    logger.info(f"[STUB] Opening app: {app_name}")
    
    # Stub implementation
    if platform.system() == "Windows":
        # TODO: Use subprocess to launch Windows apps
        pass
    elif platform.system() == "Darwin":  # macOS
        # TODO: Use 'open' command
        pass
    elif platform.system() == "Linux":
        # TODO: Use appropriate launcher
        pass
    
    return {"message": f"App '{app_name}' opened (stub)"}


async def handle_close_app(params: dict) -> dict:
    """TODO: Close an application."""
    app_name = params.get("app_name", "").strip()
    logger.info(f"[STUB] Closing app: {app_name}")
    return {"message": f"App '{app_name}' closed (stub)"}


async def handle_search(params: dict) -> dict:
    """TODO: Perform search (web or local)."""
    query = params.get("query", "")
    logger.info(f"[STUB] Searching for: {query}")
    return {"message": f"Search for '{query}' completed (stub)", "results": []}


async def handle_file_operation(params: dict) -> dict:
    """TODO: File operations (create, delete, move, etc.)."""
    operation = params.get("operation", "")
    logger.info(f"[STUB] File operation: {operation}")
    return {"message": f"File operation '{operation}' completed (stub)"}


async def handle_system_control(params: dict) -> dict:
    """TODO: System control (volume, brightness, etc.)."""
    control = params.get("control", "")
    logger.info(f"[STUB] System control: {control}")
    return {"message": f"System control '{control}' executed (stub)"}


async def handle_reminder(params: dict) -> dict:
    """TODO: Set reminder/alarm."""
    text = params.get("text", "")
    logger.info(f"[STUB] Setting reminder: {text}")
    return {"message": f"Reminder set (stub)"}


async def handle_unknown(params: dict) -> dict:
    """Fallback for unknown intents."""
    logger.warning(f"Unknown intent with params: {params}")
    return {"message": "Unknown action"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.execution_service_port,
        log_level=settings.log_level.lower()
    )

