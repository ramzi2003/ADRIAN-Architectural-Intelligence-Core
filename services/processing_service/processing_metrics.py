"""
Processing Service metrics tracking.

Tracks routing decisions, latencies, and performance metrics for the processing pipeline.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional

from services.processing_service.routing_controller import RouteType, RoutingDecision


@dataclass
class ProcessingMetrics:
    """Per-conversation processing metrics."""
    
    correlation_id: str
    intent_classification_at: float = 0.0
    routing_decision_at: float = 0.0
    response_generation_at: float = 0.0
    total_processing_at: float = 0.0
    
    intent: Optional[str] = None
    confidence: Optional[float] = None
    route_type: Optional[RouteType] = None
    route_reason: Optional[str] = None
    
    llm_used: bool = False
    direct_handler_used: bool = False
    deferred_task: bool = False
    
    errors: List[str] = field(default_factory=list)


class ProcessingServiceMetrics:
    """Aggregate metrics tracker for processing service."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._conversations: Dict[str, ProcessingMetrics] = {}
        
        # Latency history (rolling windows)
        self._latency_history: Dict[str, Deque[float]] = {
            "intent_classification": deque(maxlen=100),
            "routing_decision": deque(maxlen=100),
            "response_generation": deque(maxlen=100),
            "total_processing": deque(maxlen=100),
        }
        
        # Route decision counters
        self._route_counts: Dict[RouteType, int] = {
            RouteType.DIRECT_HANDLER: 0,
            RouteType.LLM_CALL: 0,
            RouteType.DEFERRED_TASK: 0,
        }
        
        # Intent distribution
        self._intent_counts: Dict[str, int] = {}
        
        # Error tracking
        self._error_counts: Dict[str, int] = {}
        self.total_processed: int = 0
        self.total_errors: int = 0
        
        # Recent events for debugging
        self._recent_events: Deque[dict] = deque(maxlen=50)
    
    def start_conversation(self, correlation_id: str) -> ProcessingMetrics:
        """Start tracking a new conversation."""
        with self._lock:
            metrics = ProcessingMetrics(correlation_id=correlation_id)
            self._conversations[correlation_id] = metrics
            self.total_processed += 1
            return metrics
    
    def record_intent_classification(
        self,
        correlation_id: str,
        intent: str,
        confidence: float,
        latency_ms: float
    ):
        """Record intent classification timing."""
        with self._lock:
            metrics = self._conversations.get(correlation_id)
            if metrics:
                metrics.intent_classification_at = time.perf_counter()
                metrics.intent = intent
                metrics.confidence = confidence
                self._latency_history["intent_classification"].append(latency_ms)
                
                # Track intent distribution
                self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1
                
                self._store_event(correlation_id, "intent_classified", {
                    "intent": intent,
                    "confidence": confidence,
                    "latency_ms": latency_ms
                })
    
    def record_routing_decision(
        self,
        correlation_id: str,
        decision: RoutingDecision,
        latency_ms: float
    ):
        """Record routing decision."""
        with self._lock:
            metrics = self._conversations.get(correlation_id)
            if metrics:
                metrics.routing_decision_at = time.perf_counter()
                metrics.route_type = decision.route_type
                metrics.route_reason = decision.reason
                metrics.llm_used = decision.requires_llm
                metrics.direct_handler_used = (decision.route_type == RouteType.DIRECT_HANDLER)
                metrics.deferred_task = (decision.route_type == RouteType.DEFERRED_TASK)
                
                self._latency_history["routing_decision"].append(latency_ms)
                self._route_counts[decision.route_type] = self._route_counts.get(decision.route_type, 0) + 1
                
                self._store_event(correlation_id, "routing_decided", {
                    "route_type": decision.route_type.value,
                    "reason": decision.reason,
                    "estimated_latency_ms": decision.estimated_latency_ms,
                    "latency_ms": latency_ms
                })
    
    def record_response_generation(
        self,
        correlation_id: str,
        latency_ms: float
    ):
        """Record response generation timing."""
        with self._lock:
            metrics = self._conversations.get(correlation_id)
            if metrics:
                metrics.response_generation_at = time.perf_counter()
                self._latency_history["response_generation"].append(latency_ms)
                
                self._store_event(correlation_id, "response_generated", {
                    "latency_ms": latency_ms
                })
    
    def finish_conversation(self, correlation_id: str, total_latency_ms: float):
        """Mark conversation as complete."""
        with self._lock:
            metrics = self._conversations.get(correlation_id)
            if metrics:
                metrics.total_processing_at = time.perf_counter()
                self._latency_history["total_processing"].append(total_latency_ms)
                
                self._store_event(correlation_id, "conversation_complete", {
                    "total_latency_ms": total_latency_ms,
                    "route_type": metrics.route_type.value if metrics.route_type else None
                })
                
                # Remove from active conversations
                self._conversations.pop(correlation_id, None)
    
    def record_error(self, correlation_id: str, error: str, error_type: str = "unknown"):
        """Record an error."""
        with self._lock:
            metrics = self._conversations.get(correlation_id)
            if metrics:
                metrics.errors.append(error)
            
            self.total_errors += 1
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
            
            self._store_event(correlation_id, "error", {
                "error": error,
                "error_type": error_type
            })
    
    def _store_event(self, correlation_id: str, event_type: str, data: dict):
        """Store a recent event for debugging."""
        event = {
            "correlation_id": correlation_id,
            "event_type": event_type,
            "timestamp": time.time(),
            **data
        }
        self._recent_events.appendleft(event)
    
    def get_summary(self) -> dict:
        """Get aggregated metrics summary."""
        with self._lock:
            def avg(values: Deque[float]) -> float:
                return round(sum(values) / len(values), 2) if values else 0.0
            
            summary = {
                "total_processed": self.total_processed,
                "active_conversations": len(self._conversations),
                "total_errors": self.total_errors,
                "averages": {
                    "intent_classification_ms": avg(self._latency_history["intent_classification"]),
                    "routing_decision_ms": avg(self._latency_history["routing_decision"]),
                    "response_generation_ms": avg(self._latency_history["response_generation"]),
                    "total_processing_ms": avg(self._latency_history["total_processing"]),
                },
                "route_distribution": {
                    route.value: count
                    for route, count in self._route_counts.items()
                },
                "intent_distribution": dict(self._intent_counts),
                "error_distribution": dict(self._error_counts),
                "recent_events": list(self._recent_events)[:10],  # Last 10 events
            }
            
            return summary


# Global singleton instance
_metrics_instance: Optional[ProcessingServiceMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> ProcessingServiceMetrics:
    """Get or create the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = ProcessingServiceMetrics()
    return _metrics_instance

