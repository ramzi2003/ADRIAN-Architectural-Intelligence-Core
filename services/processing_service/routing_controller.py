"""
Routing Controller for ADRIAN Processing Service.

Decides routing strategy for each utterance:
- Direct intent handler: Simple, well-defined actions (e.g., "open app")
- LLM call: Complex conversation or ambiguous requests
- Deferred task: Long-running operations that should be queued
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

from shared.logging_config import setup_logging
from shared.config import Settings, get_settings
from services.processing_service.intent_classifier import IntentPrediction


logger = setup_logging("processing-service.routing")


class RouteType(str, Enum):
    """Types of routing strategies."""
    DIRECT_HANDLER = "direct_handler"  # Fast, rule-based handler
    LLM_CALL = "llm_call"  # Requires LLM reasoning
    DEFERRED_TASK = "deferred_task"  # Long-running, should be queued


@dataclass
class RoutingDecision:
    """Represents a routing decision for an utterance."""
    route_type: RouteType
    intent: str
    confidence: float
    reason: str  # Human-readable explanation
    estimated_latency_ms: int  # Estimated processing time
    requires_llm: bool
    requires_permission: bool
    metadata: Dict[str, Any]


class RoutingController:
    """
    Middleware that decides routing strategy based on intent, confidence, and context.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._direct_handler_intents = set(self._settings.routing_direct_handler_intents)
        self._deferred_task_intents = set(self._settings.routing_deferred_task_intents)
        self._llm_required_intents = set(self._settings.routing_llm_required_intents)
        self._low_confidence_threshold = self._settings.intent_classifier_confidence_threshold
        
    def decide_route(
        self,
        text: str,
        prediction: IntentPrediction,
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """
        Decide routing strategy for an utterance.
        
        Args:
            text: User's utterance
            prediction: Intent classification result
            context: Conversation context
            
        Returns:
            RoutingDecision with strategy and metadata
        """
        intent = prediction.label
        confidence = prediction.confidence
        parameters = prediction.parameters
        
        # Low confidence -> always route to LLM for clarification
        if confidence < self._low_confidence_threshold:
            return RoutingDecision(
                route_type=RouteType.LLM_CALL,
                intent=intent,
                confidence=confidence,
                reason=f"Low confidence ({confidence:.2f}) - requires LLM reasoning",
                estimated_latency_ms=1500,
                requires_llm=True,
                requires_permission=False,
                metadata={"low_confidence": True, "original_intent": intent}
            )
        
        # Check for deferred tasks (long-running operations)
        if intent in self._deferred_task_intents:
            # Check if it's a complex task that should be deferred
            if self._should_defer_task(text, intent, parameters):
                return RoutingDecision(
                    route_type=RouteType.DEFERRED_TASK,
                    intent=intent,
                    confidence=confidence,
                    reason=f"Long-running task detected for intent '{intent}'",
                    estimated_latency_ms=5000,  # Deferred tasks take longer
                    requires_llm=False,
                    requires_permission=False,
                    metadata={"task_type": intent, "parameters": parameters}
                )
        
        # Check for direct handlers (simple, fast actions)
        if intent in self._direct_handler_intents:
            # Verify it's a simple action that doesn't need LLM
            if self._can_handle_directly(text, intent, parameters):
                return RoutingDecision(
                    route_type=RouteType.DIRECT_HANDLER,
                    intent=intent,
                    confidence=confidence,
                    reason=f"Simple '{intent}' action - can handle directly",
                    estimated_latency_ms=200,  # Very fast
                    requires_llm=False,
                    requires_permission=self._requires_permission(intent, parameters),
                    metadata={"action": parameters.get("action"), "target": parameters.get("target")}
                )
        
        # Default: route to LLM for complex reasoning
        if intent in self._llm_required_intents:
            return RoutingDecision(
                route_type=RouteType.LLM_CALL,
                intent=intent,
                confidence=confidence,
                reason=f"Intent '{intent}' requires LLM reasoning",
                estimated_latency_ms=1200,
                requires_llm=True,
                requires_permission=False,
                metadata={"intent_type": intent}
            )
        
        # Fallback: use LLM for ambiguous cases
        return RoutingDecision(
            route_type=RouteType.LLM_CALL,
            intent=intent,
            confidence=confidence,
            reason="Ambiguous request - defaulting to LLM",
            estimated_latency_ms=1500,
            requires_llm=True,
            requires_permission=False,
            metadata={"fallback": True}
        )
    
    def _can_handle_directly(self, text: str, intent: str, parameters: Dict[str, Any]) -> bool:
        """
        Check if an intent can be handled directly without LLM.
        
        Simple system_control actions like "open app" can be handled directly.
        Complex requests like "open the app I was using yesterday" need LLM.
        """
        if intent != "system_control":
            return False
        
        action = parameters.get("action", "").lower()
        target = parameters.get("target", "").lower()
        
        # Simple actions: open, close, start, stop
        simple_actions = {"open", "close", "start", "stop", "launch"}
        if action in simple_actions and target:
            # Check if target is simple (not a complex description)
            if len(target.split()) <= 3:  # Simple target like "spotify" or "web browser"
                return True
        
        # Complex requests need LLM
        return False
    
    def _should_defer_task(self, text: str, intent: str, parameters: Dict[str, Any]) -> bool:
        """
        Check if a task should be deferred (queued for later execution).
        
        Simple reminders can be handled immediately.
        Complex task management operations should be deferred.
        """
        if intent != "task_management":
            return False
        
        # Check for keywords that indicate long-running operations
        text_lower = text.lower()
        defer_keywords = ["schedule", "recurring", "every", "daily", "weekly", "complex"]
        
        if any(keyword in text_lower for keyword in defer_keywords):
            return True
        
        # Simple reminders/todos can be handled immediately
        return False
    
    def _requires_permission(self, intent: str, parameters: Dict[str, Any]) -> bool:
        """Check if an action requires user permission."""
        dangerous_actions = {"delete", "uninstall", "shutdown", "restart", "format"}
        action = parameters.get("action", "").lower()
        
        if action in dangerous_actions:
            return True
        
        # System control intents with dangerous actions
        if intent == "system_control" and action in dangerous_actions:
            return True
        
        return False


# Singleton instance
_routing_controller: Optional[RoutingController] = None


def get_routing_controller(settings: Optional[Settings] = None) -> RoutingController:
    """Get or create the routing controller singleton."""
    global _routing_controller
    if _routing_controller is None:
        _routing_controller = RoutingController(settings)
    return _routing_controller

