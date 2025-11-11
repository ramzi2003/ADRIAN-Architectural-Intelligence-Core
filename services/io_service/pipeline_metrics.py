"""Voice pipeline metrics tracking for ADRIAN IO Service."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional


@dataclass
class ConversationMetrics:
    """Per-conversation timing data for the voice pipeline."""

    hotword_at: float = field(default_factory=time.perf_counter)
    stt_completed_at: float = 0.0
    response_received_at: float = 0.0
    tts_started_at: float = 0.0
    tts_completed_at: float = 0.0
    transcript: Optional[str] = None
    response_text: Optional[str] = None
    confidence: Optional[float] = None
    retries: int = 0
    errors: List[str] = field(default_factory=list)

    def stage_latency_ms(self, stage_time: float) -> Optional[float]:
        if stage_time <= 0 or self.hotword_at <= 0:
            return None
        return (stage_time - self.hotword_at) * 1000


class VoicePipelineMetrics:
    """Aggregate tracker for voice pipeline timings and health."""

    def __init__(self, latency_target_ms: int = 2000):
        self.latency_target_ms = latency_target_ms
        self._lock = threading.Lock()
        self._conversations: Dict[str, ConversationMetrics] = {}
        self._latency_history: Dict[str, Deque[float]] = {
            "hotword_to_stt": deque(maxlen=50),
            "hotword_to_response": deque(maxlen=50),
            "hotword_to_tts_start": deque(maxlen=50),
            "hotword_to_tts_end": deque(maxlen=50),
            "tts_duration": deque(maxlen=50),
        }
        self._events: Deque[dict] = deque(maxlen=25)
        self.total_conversations: int = 0
        self.total_interrupts: int = 0
        self.total_errors: int = 0

    def _store_event(self, conversation_id: str, stage: str, latency_ms: Optional[float], extra: Optional[dict] = None):
        event = {
            "conversation_id": conversation_id,
            "stage": stage,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }
        if extra:
            event.update(extra)
        self._events.appendleft(event)

    def record_stage(self, conversation_id: str, stage: str, **kwargs):
        """Record a pipeline stage for a given conversation."""
        now = time.perf_counter()
        with self._lock:
            metrics = self._conversations.setdefault(conversation_id, ConversationMetrics())
            latency_ms: Optional[float] = None

            if stage == "hotword":
                metrics.hotword_at = now
                metrics.retries = 0
                metrics.errors.clear()
                metrics.transcript = None
                metrics.response_text = None
                metrics.confidence = None
                self.total_conversations += 1
            elif stage == "stt_completed":
                metrics.stt_completed_at = now
                metrics.transcript = kwargs.get("transcript")
                metrics.confidence = kwargs.get("confidence")
                metrics.retries = kwargs.get("retries", metrics.retries)
                latency_ms = metrics.stage_latency_ms(metrics.stt_completed_at)
                if latency_ms is not None:
                    self._latency_history["hotword_to_stt"].append(latency_ms)
            elif stage == "response_received":
                metrics.response_received_at = now
                metrics.response_text = kwargs.get("response_text")
                latency_ms = metrics.stage_latency_ms(metrics.response_received_at)
                if latency_ms is not None:
                    self._latency_history["hotword_to_response"].append(latency_ms)
            elif stage == "tts_started":
                metrics.tts_started_at = now
                latency_ms = metrics.stage_latency_ms(metrics.tts_started_at)
                if latency_ms is not None:
                    self._latency_history["hotword_to_tts_start"].append(latency_ms)
            elif stage == "tts_completed":
                metrics.tts_completed_at = now
                latency_ms = metrics.stage_latency_ms(metrics.tts_completed_at)
                if latency_ms is not None:
                    self._latency_history["hotword_to_tts_end"].append(latency_ms)
                if metrics.tts_started_at > 0:
                    tts_duration_ms = (metrics.tts_completed_at - metrics.tts_started_at) * 1000
                    self._latency_history["tts_duration"].append(tts_duration_ms)
                # Conversation cycle finished; remove from active map
                self._conversations.pop(conversation_id, None)
            elif stage == "interrupted":
                self.total_interrupts += 1
            elif stage == "retry":
                metrics.retries = metrics.retries + 1
            else:
                raise ValueError(f"Unknown stage '{stage}'")

            self._store_event(conversation_id, stage, latency_ms, kwargs if kwargs else None)

    def record_error(self, conversation_id: str, error: str):
        """Record an error for a given conversation."""
        with self._lock:
            metrics = self._conversations.setdefault(conversation_id, ConversationMetrics())
            metrics.errors.append(error)
            self.total_errors += 1
            self._store_event(conversation_id, "error", None, {"error": error})

    def get_summary(self) -> dict:
        """Return aggregated metrics summary."""
        with self._lock:
            def avg(values: Deque[float]) -> float:
                return round(sum(values) / len(values), 2) if values else 0.0

            summary = {
                "latency_target_ms": self.latency_target_ms,
                "averages": {
                    "hotword_to_stt": avg(self._latency_history["hotword_to_stt"]),
                    "hotword_to_response": avg(self._latency_history["hotword_to_response"]),
                    "hotword_to_tts_start": avg(self._latency_history["hotword_to_tts_start"]),
                    "hotword_to_tts_end": avg(self._latency_history["hotword_to_tts_end"]),
                    "tts_duration": avg(self._latency_history["tts_duration"]),
                },
                "history": list(self._events),
                "active_conversations": len(self._conversations),
                "total_conversations": self.total_conversations,
                "total_interrupts": self.total_interrupts,
                "total_errors": self.total_errors,
            }

            # Flag if current averages exceed target
            summary["alerts"] = []
            if summary["averages"]["hotword_to_response"] and summary["averages"]["hotword_to_response"] > self.latency_target_ms:
                summary["alerts"].append(
                    {
                        "type": "latency",
                        "message": f"Average hotword→response latency {summary['averages']['hotword_to_response']}ms exceeds target {self.latency_target_ms}ms",
                    }
                )

            return summary


