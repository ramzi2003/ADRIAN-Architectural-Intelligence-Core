"""
Message schemas for inter-service communication in ADRIAN.
All events passed through Redis Pub/Sub use these schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """Types of messages exchanged between services."""
    UTTERANCE = "utterance"
    ACTION_SPEC = "action_spec"
    ACTION_RESULT = "action_result"
    RESPONSE_TEXT = "response_text"
    MEMORY_QUERY = "memory_query"
    MEMORY_RESULT = "memory_result"
    SECURITY_CHECK = "security_check"
    SECURITY_RESULT = "security_result"
    TTS_EVENT = "tts_event"


class BaseMessage(BaseModel):
    """Base message with correlation tracking."""
    message_id: str = Field(..., description="Unique message ID")
    correlation_id: str = Field(..., description="Request correlation ID for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: MessageType


class UtteranceEvent(BaseMessage):
    """Event emitted by io-service when user speaks or types."""
    message_type: MessageType = MessageType.UTTERANCE
    text: str = Field(..., description="Transcribed or typed text from user")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(..., description="'voice' or 'text'")
    user_id: Optional[str] = Field(None, description="Identified speaker/user")


class ActionSpec(BaseMessage):
    """Specification of an action to be executed."""
    message_type: MessageType = MessageType.ACTION_SPEC
    intent: str = Field(..., description="Classified intent (e.g., 'open_app', 'search', 'reminder')")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    requires_permission: bool = Field(default=False, description="Whether action needs security check")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ActionResult(BaseMessage):
    """Result of an executed action."""
    message_type: MessageType = MessageType.ACTION_RESULT
    action_id: str = Field(..., description="ID of the action that was executed")
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ResponseText(BaseMessage):
    """Text response to be spoken/displayed to user."""
    message_type: MessageType = MessageType.RESPONSE_TEXT
    text: str = Field(..., description="Response text from ADRIAN")
    should_speak: bool = Field(default=True, description="Whether to use TTS")
    emotion: Optional[str] = Field(None, description="Suggested emotional tone")


class MemoryQuery(BaseMessage):
    """Query to memory service for semantic search or retrieval."""
    message_type: MessageType = MessageType.MEMORY_QUERY
    query_text: str
    limit: int = Field(default=5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class MemoryResult(BaseMessage):
    """Result from memory service."""
    message_type: MessageType = MessageType.MEMORY_RESULT
    query_id: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    

class SecurityCheck(BaseMessage):
    """Request to security service to verify permissions."""
    message_type: MessageType = MessageType.SECURITY_CHECK
    action_intent: str
    user_id: Optional[str] = None
    requires_auth: bool = True


class SecurityResult(BaseMessage):
    """Result from security service."""
    message_type: MessageType = MessageType.SECURITY_RESULT
    check_id: str
    approved: bool
    reason: Optional[str] = None


class TTSPlaybackEvent(BaseMessage):
    """Event emitted by output-service when TTS playback starts/ends."""

    message_type: MessageType = MessageType.TTS_EVENT
    event: str = Field(..., description="'start', 'end', or 'error'")
    conversation_id: Optional[str] = Field(None, description="Conversation correlation ID")
    text: Optional[str] = Field(None, description="Text associated with playback")
    duration_ms: Optional[float] = Field(None, description="Playback duration in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Health check response
class HealthResponse(BaseModel):
    """Standard health check response for all services."""
    service_name: str
    status: str = "healthy"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dependencies: Dict[str, str] = Field(default_factory=dict)

