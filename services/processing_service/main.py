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
import time
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
from services.processing_service.routing_controller import (
    RoutingController,
    RouteType,
    RoutingDecision,
    get_routing_controller,
)
from services.processing_service.processing_metrics import (
    ProcessingServiceMetrics,
    get_metrics,
)
from services.processing_service.intent_handlers import (
    handle_intent,
    HandlerResponse,
    get_handler,
)
from services.processing_service.personality_wit import (
    rewrite_with_personality,
    TonePreset,
    get_personality_rewriter,
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
_routing_controller: Optional[RoutingController] = None
_processing_metrics: Optional[ProcessingServiceMetrics] = None


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
    
    # Initialize routing controller
    global _routing_controller
    _routing_controller = get_routing_controller(settings)
    logger.info("Routing controller initialized")
    
    # Initialize processing metrics
    global _processing_metrics
    _processing_metrics = get_metrics()
    logger.info("Processing metrics initialized")
    
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


@app.get("/metrics")
async def get_processing_metrics():
    """Get processing service metrics summary."""
    if _processing_metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized")
    return _processing_metrics.get_summary()


@app.post("/personality/tone")
async def set_tone_override(
    tone: str,
    duration_seconds: Optional[int] = None
):
    """
    Set a temporary tone override for subsequent responses.
    
    Args:
        tone: Tone preset name (jarvis, formal, minimal, friendly, sarcastic, professional)
        duration_seconds: Optional duration for override (None = until next override)
    
    Returns:
        Confirmation message
    """
    try:
        tone_preset = TonePreset(tone.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tone: {tone}. Must be one of: {[t.value for t in TonePreset]}"
        )
    
    # Store override in a simple in-memory cache (in production, use Redis or database)
    # For now, this is a stub - actual implementation would store per-user/session
    logger.info(f"Tone override set to: {tone_preset.value} (duration: {duration_seconds}s)")
    
    return {
        "status": "success",
        "tone": tone_preset.value,
        "duration_seconds": duration_seconds,
        "message": f"Tone override set to {tone_preset.value}"
    }


@app.get("/personality/tone")
async def get_current_tone():
    """Get current personality tone configuration."""
    rewriter = get_personality_rewriter(settings)
    current_tone = rewriter._default_config.tone
    
    return {
        "tone": current_tone.value,
        "available_tones": [t.value for t in TonePreset],
        "config": {
            "use_contractions": rewriter._default_config.use_contractions,
            "add_sir": rewriter._default_config.add_sir,
            "wit_level": rewriter._default_config.wit_level,
            "formality_level": rewriter._default_config.formality_level,
        }
    }


@app.post("/personality/config")
async def update_personality_config(
    use_contractions: Optional[bool] = None,
    add_sir: Optional[bool] = None,
    wit_level: Optional[int] = None,
    formality_level: Optional[int] = None,
):
    """
    Update personality configuration.
    
    Args:
        use_contractions: Whether to use contractions
        add_sir: Whether to add "Sir" to responses
        wit_level: Wit level (0-3)
        formality_level: Formality level (0-4)
    
    Returns:
        Updated configuration
    """
    rewriter = get_personality_rewriter(settings)
    
    if use_contractions is not None:
        rewriter._default_config.use_contractions = use_contractions
    if add_sir is not None:
        rewriter._default_config.add_sir = add_sir
    if wit_level is not None:
        rewriter._default_config.wit_level = max(0, min(3, wit_level))
    if formality_level is not None:
        rewriter._default_config.formality_level = max(0, min(4, formality_level))
    
    return {
        "status": "success",
        "config": {
            "tone": rewriter._default_config.tone.value,
            "use_contractions": rewriter._default_config.use_contractions,
            "add_sir": rewriter._default_config.add_sir,
            "wit_level": rewriter._default_config.wit_level,
            "formality_level": rewriter._default_config.formality_level,
        }
    }


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
    Main processing pipeline with routing:
    1. Query memory for context
    2. Classify intent
    3. Route decision (direct handler vs LLM vs deferred)
    4. Execute based on route
    5. Generate response with personality
    6. Emit action_spec and response_text
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    start_time = time.perf_counter()
    redis_client = await get_redis_client()
    
    # Initialize metrics tracking
    if _processing_metrics:
        _processing_metrics.start_conversation(correlation_id)
    
    try:
        # Step 1: Query memory service for context (STUB)
        # TODO: Make actual call to memory-service
        context = await query_memory_context(text, correlation_id)
        
        # Step 2: Classify intent
        intent_start = time.perf_counter()
        prediction = await classify_intent(text, context)
        intent_latency_ms = (time.perf_counter() - intent_start) * 1000
        
        if _processing_metrics:
            _processing_metrics.record_intent_classification(
                correlation_id,
                prediction.label,
                prediction.confidence,
                intent_latency_ms
            )
        
        intent = prediction.label
        parameters = prediction.parameters
        
        # Step 3: Routing decision
        routing_start = time.perf_counter()
        if _routing_controller is None:
            raise RuntimeError("Routing controller not initialized")
        
        routing_decision = _routing_controller.decide_route(text, prediction, context)
        routing_latency_ms = (time.perf_counter() - routing_start) * 1000
        
        if _processing_metrics:
            _processing_metrics.record_routing_decision(
                correlation_id,
                routing_decision,
                routing_latency_ms
            )
        
        logger.info(
            "Routing decision: %s for intent '%s' (reason: %s, estimated: %dms)",
            routing_decision.route_type.value,
            routing_decision.intent,
            routing_decision.reason,
            routing_decision.estimated_latency_ms
        )
        
        # Step 4: Execute based on route
        response_start = time.perf_counter()
        handler_response = await execute_route(
            routing_decision,
            text,
            intent,
            parameters,
            context
        )
        response_latency_ms = (time.perf_counter() - response_start) * 1000
        
        if _processing_metrics:
            _processing_metrics.record_response_generation(
                correlation_id,
                response_latency_ms
            )
        
        # Step 5: Apply personality/wit module
        if settings.personality_enable_rewrite:
            # Check for tone override in context or metadata
            tone_override = None
            config_override = None
            
            # Check metadata for tone override
            if handler_response.metadata:
                tone_str = handler_response.metadata.get("tone_override")
                if tone_str:
                    try:
                        tone_override = TonePreset(tone_str.lower())
                    except ValueError:
                        logger.warning(f"Invalid tone override: {tone_str}")
                
                # Check for config overrides
                if "personality_config" in handler_response.metadata:
                    config_override = handler_response.metadata["personality_config"]
            
            response_text = await rewrite_with_personality(
                handler_response.text,
                tone_override=tone_override,
                config_override=config_override,
                settings=settings
            )
        else:
            response_text = handler_response.text
        
        # Step 6: Emit action_spec if needed
        # Use action_spec from handler if available, otherwise create from routing decision
        if handler_response.action_spec:
            # Handler provided action_spec dict - extract fields
            handler_action_spec = handler_response.action_spec
            action_spec = ActionSpec(
                message_id=str(uuid.uuid4()),
                correlation_id=correlation_id,
                intent=handler_action_spec.get("intent", intent),
                parameters={
                    **parameters,  # Start with original parameters
                    **{k: v for k, v in handler_action_spec.items() 
                       if k not in ["intent", "requires_permission"]}  # Merge handler params
                },
                requires_permission=handler_action_spec.get("requires_permission", routing_decision.requires_permission),
                confidence=prediction.confidence
            )
            await redis_client.publish("action_specs", action_spec)
            logger.info(
                "Published action_spec: %s (confidence=%.2f, route=%s)",
                intent,
                prediction.confidence,
                routing_decision.route_type.value,
            )
        elif intent != "conversation" and routing_decision.route_type != RouteType.DEFERRED_TASK:
            # Fallback: create action_spec from routing decision
            action_spec = ActionSpec(
                message_id=str(uuid.uuid4()),
                correlation_id=correlation_id,
                intent=intent,
                parameters=parameters,
                requires_permission=routing_decision.requires_permission,
                confidence=prediction.confidence
            )
            await redis_client.publish("action_specs", action_spec)
            logger.info(
                "Published action_spec: %s (confidence=%.2f, route=%s)",
                intent,
                prediction.confidence,
                routing_decision.route_type.value,
            )
        
        # Step 7: Emit response_text
        # Extract emotion from handler metadata if available
        emotion = handler_response.metadata.get("emotion")
        response = ResponseText(
            message_id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            text=response_text,
            should_speak=True,
            emotion=emotion
        )
        await redis_client.publish("responses", response)
        logger.info(f"Published response: '{response_text}'")
        
        # Record completion
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        if _processing_metrics:
            _processing_metrics.finish_conversation(correlation_id, total_latency_ms)
        
        return {
            "intent": intent,
            "response": response_text,
            "correlation_id": correlation_id,
            "route_type": routing_decision.route_type.value,
            "route_reason": routing_decision.reason,
            "total_latency_ms": total_latency_ms,
            "handler_metadata": handler_response.metadata,
            "action_spec_created": handler_response.action_spec is not None
        }
    
    except Exception as e:
        logger.exception(f"Error processing utterance: {e}")
        if _processing_metrics:
            _processing_metrics.record_error(correlation_id, str(e), type(e).__name__)
        raise


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


async def execute_route(
    decision: RoutingDecision,
    text: str,
    intent: str,
    parameters: dict,
    context: dict
) -> HandlerResponse:
    """
    Execute the routing decision.
    
    Routes to:
    - Direct handler: Fast rule-based response using intent handlers
    - LLM call: Full LLM reasoning (may still use handlers for structure)
    - Deferred task: Queue for later execution
    
    Returns:
        HandlerResponse with normalized output
    """
    if decision.route_type == RouteType.DIRECT_HANDLER:
        # Use intent handler for direct responses
        try:
            handler_response = await handle_intent(intent, text, parameters, context, settings)
            return handler_response
        except ValueError:
            # No handler found, fallback to LLM
            logger.warning(f"No handler for intent '{intent}', falling back to LLM")
            response_text = await generate_response(text, intent, context)
            return HandlerResponse(
                text=response_text,
                action_spec=None,
                metadata={"intent": intent, "fallback": True}
            )
    
    elif decision.route_type == RouteType.LLM_CALL:
        # For LLM calls, try handler first for structure, then enhance with LLM
        # If handler exists, use it; otherwise use LLM directly
        handler = get_handler(intent, settings)
        if handler:
            # Use handler to get structured response, then enhance with LLM if needed
            handler_response = await handler.handle(text, parameters, context)
            # For conversation intents, enhance with LLM
            if intent == "conversation" or handler_response.metadata.get("requires_llm"):
                llm_text = await generate_response(text, intent, context)
                handler_response.text = llm_text
            return handler_response
        else:
            # No handler, use LLM directly
            response_text = await generate_response(text, intent, context)
            return HandlerResponse(
                text=response_text,
                action_spec=None,
                metadata={"intent": intent, "llm_generated": True}
            )
    
    elif decision.route_type == RouteType.DEFERRED_TASK:
        # For deferred tasks, use handler to create structured response
        # In production, this would queue the task
        logger.info("Deferred task detected - executing immediately (TODO: implement queue)")
        try:
            handler_response = await handle_intent(intent, text, parameters, context, settings)
            handler_response.metadata["deferred"] = True
            return handler_response
        except ValueError:
            # Fallback to LLM
            response_text = await generate_response(text, intent, context)
            return HandlerResponse(
                text=response_text,
                action_spec=None,
                metadata={"intent": intent, "deferred": True, "fallback": True}
            )
    
    else:
        # Fallback to LLM
        logger.warning(f"Unknown route type {decision.route_type}, falling back to LLM")
        response_text = await generate_response(text, intent, context)
        return HandlerResponse(
            text=response_text,
            action_spec=None,
            metadata={"intent": intent, "fallback": True}
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
    
    This function is kept for backward compatibility.
    Use rewrite_with_personality() directly for more control.
    """
    if settings.personality_enable_rewrite:
        return await rewrite_with_personality(response, settings=settings)
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

