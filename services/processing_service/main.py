"""
Processing Service - NLP and Reasoning Layer
Handles intent classification, LLM interaction, and personality injection.
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import uuid
import httpx
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.schemas import (
    UtteranceEvent, ActionSpec, ResponseText, 
    HealthResponse, MemoryQuery
)
from shared.redis_client import get_redis_client
from shared.logging_config import setup_logging

from services.processing_service.llm_runtime import (
    LLMRuntime,
    LLMRuntimeError,
    get_runtime,
)
from services.processing_service.intent_classifier import (
    IntentClassifier,
    IntentClassifierError,
    IntentPrediction,
    get_classifier,
)

# Setup logging
logger = setup_logging("processing-service")
settings = get_settings()


# Request models
class ProcessRequest(BaseModel):
    """Request model for synchronous processing."""
    text: str
    user_id: str = "default_user"


# Background task flag
_listening = False
_llm_runtime: Optional[LLMRuntime] = None
_intent_classifier: Optional[IntentClassifier] = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _listening
    logger.info("Starting Processing Service...")
    
    # Connect to Redis
    redis_client = await get_redis_client()
    logger.info("Connected to Redis")

    # Initialize local LLM runtime if configured
    if settings.llm_provider == "llama_cpp":
        try:
            global _llm_runtime
            _llm_runtime = await get_runtime()
            await _llm_runtime.warmup()
        except LLMRuntimeError as exc:
            logger.error("Failed to initialize local LLM runtime: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected LLM initialization error: %s", exc)
    else:
        _llm_runtime = None

    # Initialize intent classifier (optional)
    try:
        global _intent_classifier
        _intent_classifier = await get_classifier()
    except IntentClassifierError as exc:
        logger.error("Failed to initialize intent classifier: %s", exc)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected intent classifier error: %s", exc)
    
    # Start background listener for utterances
    _listening = True
    asyncio.create_task(listen_for_utterances())
    
    print("\n" + "="*60)
    print("✅ Processing Service ready - Listening for utterances...")
    print("="*60 + "\n")
    
    # TODO: Initialize connection to Ollama
    # TODO: Initialize Gemini client (optional)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Processing Service...")
    _listening = False
    await redis_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="ADRIAN Processing Service",
    description="NLP and Reasoning Layer",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    ollama_status = await check_ollama_connection()
    llm_status = "not_configured"
    classifier_status = "not_loaded"
    if settings.llm_provider == "llama_cpp":
        runtime = _llm_runtime
        if runtime and runtime.is_loaded:
            llm_status = "loaded"
        else:
            llm_status = "loading_failed" if runtime else "not_loaded"
    else:
        llm_status = settings.llm_provider

    classifier = _intent_classifier
    if classifier and classifier.is_loaded:
        classifier_status = "loaded"
    elif classifier is None:
        classifier_status = "not_initialized"
    else:
        classifier_status = "loading_failed"
    
    return HealthResponse(
        service_name="processing-service",
        status="healthy",
        dependencies={
            "redis": "connected",
            "ollama": ollama_status,
            "llm_runtime": llm_status,
            "intent_classifier": classifier_status,
            "gemini": "not_configured" if not settings.gemini_api_key else "ready"
        }
    )


@app.post("/process")
async def process_text(request: ProcessRequest):
    """
    Synchronous endpoint to process text and return intent/response.
    Useful for testing without going through Redis.
    """
    try:
        result = await process_utterance(request.text, request.user_id)
        return result
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def listen_for_utterances():
    """
    Background task that listens to utterance events from Redis.
    """
    logger.info("Started listening for utterances...")
    redis_client = await get_redis_client()
    
    async def handle_utterance(data: dict):
        """Process incoming utterance."""
        try:
            utterance = UtteranceEvent(**data)
            logger.info(f"Received utterance: '{utterance.text}' from {utterance.source}")
            
            # Process the utterance
            await process_utterance(
                utterance.text, 
                utterance.user_id or "default_user",
                utterance.correlation_id
            )
        except Exception as e:
            logger.error(f"Error handling utterance: {e}")
    
    # Subscribe and listen
    await redis_client.subscribe("utterances", handle_utterance)


async def process_utterance(text: str, user_id: str, correlation_id: Optional[str] = None):
    """
    Main processing pipeline:
    1. Query memory for context
    2. Classify intent
    3. Generate response with personality
    4. Emit action_spec and response_text
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    redis_client = await get_redis_client()
    
    # Step 1: Query memory service for context (STUB)
    # TODO: Make actual call to memory-service
    context = await query_memory_context(text, correlation_id)
    
    # Step 2: Classify intent
    prediction = await classify_intent(text, context)
    intent = prediction.label
    parameters = prediction.parameters
    
    # Step 3: Generate response (STUB)
    response_text = await generate_response(text, intent, context)
    
    # Step 4: Apply personality/wit module (STUB)
    response_text = await apply_personality(response_text, intent)
    
    # Step 5: Emit action_spec if needed
    if intent != "conversation":
        action_spec = ActionSpec(
            message_id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            intent=intent,
            parameters=parameters,
            requires_permission=intent in ["delete_file", "shutdown", "install_app"],
            confidence=prediction.confidence
        )
        await redis_client.publish("action_specs", action_spec)
        logger.info(
            "Published action_spec: %s (confidence=%.2f, classifier=%s)",
            intent,
            prediction.confidence,
            "trained" if prediction.used_classifier else "heuristic",
        )
    
    # Step 6: Emit response_text
    response = ResponseText(
        message_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        text=response_text,
        should_speak=True
    )
    await redis_client.publish("responses", response)
    logger.info(f"Published response: '{response_text}'")
    
    return {
        "intent": intent,
        "response": response_text,
        "correlation_id": correlation_id
    }


async def query_memory_context(text: str, correlation_id: str) -> dict:
    """
    Query memory service for relevant context.
    TODO: Implement actual HTTP call to memory-service.
    """
    logger.info(f"[STUB] Querying memory for context: '{text}'")
    return {"context": "stub_context", "history": []}


async def classify_intent(text: str, context: dict) -> IntentPrediction:
    """
    Determine the user's intent using the fine-tuned classifier when available.
    Falls back to heuristic rules on failure.
    """
    classifier = _intent_classifier
    if classifier and classifier.is_loaded:
        prediction = await classifier.predict(text, context)
    else:
        logger.debug("Intent classifier unavailable; using heuristic rules.")
        fallback_classifier = await get_classifier()
        prediction = await fallback_classifier.predict(text, context)

    threshold = settings.intent_classifier_confidence_threshold
    fallback_route = settings.intent_classifier_low_confidence_route or "conversation"

    final_label = prediction.label
    final_parameters = prediction.parameters
    if prediction.confidence < threshold and final_label != fallback_route:
        logger.info(
            "Intent confidence %.2f below threshold %.2f for '%s'; rerouting to %s.",
            prediction.confidence,
            threshold,
            final_label,
            fallback_route,
        )
        final_label = fallback_route
        final_parameters = {}
        rerouted = IntentPrediction(
            label=final_label,
            confidence=prediction.confidence,
            parameters=final_parameters,
            used_classifier=prediction.used_classifier,
        )
        return rerouted

    return IntentPrediction(
        label=final_label,
        confidence=prediction.confidence,
        parameters=final_parameters,
        used_classifier=prediction.used_classifier,
    )


async def generate_response(text: str, intent: str, context: dict) -> str:
    """
    Generate response text using LLM (Ollama or Gemini).
    TODO: Implement actual LLM call.
    """
    logger.info(f"[STUB] Generating response for intent: {intent}")
    llm_provider = settings.llm_provider.lower()

    if llm_provider == "llama_cpp":
        prompt = build_structured_prompt(text, intent, context)
        runtime = _llm_runtime
        if runtime is None:
            logger.warning("LLM runtime not available; falling back to stub.")
        else:
            try:
                generated = await runtime.generate(prompt)
                return generated.strip()
            except LLMRuntimeError as exc:
                logger.error("Local LLM generation failed: %s", exc)
            except Exception as exc:  # pragma: no cover
                logger.exception("Unexpected LLM error: %s", exc)

    if llm_provider == "ollama":
        try:
            response = await call_ollama(text, context)
            return response
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}, falling back to echo response")

    # Echo response for testing - confirms it heard you correctly
    text_lower = text.lower()
    if "hello" in text_lower or "hi" in text_lower:
        return "Hello, Sir. How may I assist you?"
    elif "how are you" in text_lower:
        return "I'm functioning perfectly, Sir. Thank you for asking."
    elif "what" in text_lower and "time" in text_lower:
        from datetime import datetime
        return f"The current time is {datetime.now().strftime('%I:%M %p')}, Sir."
    else:
        return f"I heard you say: {text}. At your service, Sir."


async def apply_personality(response: str, intent: str) -> str:
    """
    Apply personality/wit module to make response more natural and Jarvis-like.
    TODO: Implement personality layer.
    """
    logger.info("[STUB] Applying personality layer")
    
    # Response already has personality, just return it
    # (Avoid double-prefixing since generate_response already adds it)
    return response


def build_structured_prompt(text: str, intent: str, context: dict) -> str:
    """
    Compose a system-style prompt for the local LLM with minimal formatting.
    """
    persona = settings.llm_persona_default
    conversation_history = context.get("history", [])
    history_blocks = []
    for turn in conversation_history[-3:]:
        speaker = turn.get("speaker", "User")
        utterance = turn.get("text") or ""
        history_blocks.append(f"{speaker}: {utterance}")
    history_text = "\n".join(history_blocks) if history_blocks else "None"

    return (
        "You are ADRIAN, an efficient desktop AI assistant with a Jarvis-like tone.\n"
        f"Persona preset: {persona}.\n"
        "Follow system policy: be concise, decisive, and helpful. "
        "If the user asks for time, include the current system time.\n"
        f"Detected intent: {intent}.\n"
        f"Conversation history:\n{history_text}\n"
        "User request:\n"
        f"{text}\n"
        "Respond with a single concise paragraph.\n"
    )


async def call_ollama(prompt: str, context: dict) -> str:
    """
    Call Ollama API for LLM inference.
    TODO: Implement full Ollama integration with proper prompting.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                raise Exception(f"Ollama returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        raise


async def check_ollama_connection() -> str:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            if response.status_code == 200:
                return "connected"
            else:
                return "unreachable"
    except Exception:
        return "unreachable"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.processing_service_port,
        log_level=settings.log_level.lower()
    )

