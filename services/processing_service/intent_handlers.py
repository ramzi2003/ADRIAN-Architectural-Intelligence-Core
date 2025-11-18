"""
Intent Handlers for ADRIAN Processing Service.

Stub implementations for priority intents:
- SystemControl: Handle system/app control actions
- Search: Handle search queries
- TaskManagement: Handle reminders, todos, scheduling
- Conversation: Handle general conversation and Q&A

All handlers return normalized response payloads for TTS/UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

from shared.logging_config import setup_logging
from shared.config import Settings, get_settings


logger = setup_logging("processing-service.handlers")


@dataclass
class HandlerResponse:
    """
    Normalized response payload from intent handlers.
    All handlers return this structure for consistent TTS/UI output.
    """
    text: str  # Response text for TTS/display
    action_spec: Optional[Dict[str, Any]] = None  # Action specification if action needed
    metadata: Dict[str, Any] = None  # Additional metadata (emotion, context, etc.)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SystemControlHandler:
    """Handler for system control intents (open/close apps, system actions)."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
    
    async def handle(
        self,
        text: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> HandlerResponse:
        """
        Handle system control intent.
        
        Args:
            text: Original user utterance
            parameters: Extracted parameters (action, target, etc.)
            context: Conversation context
            
        Returns:
            HandlerResponse with normalized output
        """
        action = parameters.get("action", "").lower()
        target = parameters.get("target", "").strip()
        
        # Determine response text based on action
        if action == "open":
            if target:
                response_text = f"Opening {target}, Sir."
                action_spec = {
                    "intent": "system_control",
                    "action": "open",
                    "target": target,
                    "requires_permission": False,
                }
            else:
                response_text = "I need to know what you'd like me to open, Sir."
                action_spec = None
        
        elif action == "close":
            if target:
                response_text = f"Closing {target}, Sir."
                action_spec = {
                    "intent": "system_control",
                    "action": "close",
                    "target": target,
                    "requires_permission": False,
                }
            else:
                response_text = "I need to know what you'd like me to close, Sir."
                action_spec = None
        
        elif action == "start" or action == "launch":
            if target:
                response_text = f"Launching {target}, Sir."
                action_spec = {
                    "intent": "system_control",
                    "action": "start",
                    "target": target,
                    "requires_permission": False,
                }
            else:
                response_text = "I need to know what you'd like me to launch, Sir."
                action_spec = None
        
        elif action == "stop" or action == "quit":
            if target:
                response_text = f"Stopping {target}, Sir."
                action_spec = {
                    "intent": "system_control",
                    "action": "stop",
                    "target": target,
                    "requires_permission": False,
                }
            else:
                response_text = "I need to know what you'd like me to stop, Sir."
                action_spec = None
        
        else:
            # Generic system control action
            response_text = f"Executing {action} for {target}, Sir." if target else f"Executing {action}, Sir."
            action_spec = {
                "intent": "system_control",
                "action": action,
                "target": target,
                "requires_permission": self._requires_permission(action),
            }
        
        return HandlerResponse(
            text=response_text,
            action_spec=action_spec,
            metadata={
                "intent": "system_control",
                "action": action,
                "target": target,
                "emotion": "professional",
            }
        )
    
    def _requires_permission(self, action: str) -> bool:
        """Check if action requires user permission."""
        dangerous_actions = {"delete", "uninstall", "shutdown", "restart", "format", "remove"}
        return action.lower() in dangerous_actions


class SearchHandler:
    """Handler for search intents (web search, file search, information queries)."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
    
    async def handle(
        self,
        text: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> HandlerResponse:
        """
        Handle search intent.
        
        Args:
            text: Original user utterance
            parameters: Extracted parameters (query, search_type, etc.)
            context: Conversation context
            
        Returns:
            HandlerResponse with normalized output
        """
        query = parameters.get("query", text)
        search_type = parameters.get("search_type", "general")  # general, file, web, local
        
        # Stub implementation - in production, this would call search service
        if search_type == "file":
            response_text = f"Searching for files matching '{query}', Sir."
        elif search_type == "web":
            response_text = f"Searching the web for '{query}', Sir."
        else:
            response_text = f"Searching for '{query}', Sir."
        
        action_spec = {
            "intent": "search",
            "query": query,
            "search_type": search_type,
            "requires_permission": False,
        }
        
        return HandlerResponse(
            text=response_text,
            action_spec=action_spec,
            metadata={
                "intent": "search",
                "query": query,
                "search_type": search_type,
                "emotion": "helpful",
            }
        )


class TaskManagementHandler:
    """Handler for task management intents (reminders, todos, scheduling)."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
    
    async def handle(
        self,
        text: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> HandlerResponse:
        """
        Handle task management intent.
        
        Args:
            text: Original user utterance
            parameters: Extracted parameters (task, due_date, priority, etc.)
            context: Conversation context
            
        Returns:
            HandlerResponse with normalized output
        """
        task_text = parameters.get("text", text)
        task_type = parameters.get("task_type", "reminder")  # reminder, todo, schedule
        due_date = parameters.get("due_date")
        priority = parameters.get("priority", "normal")
        
        # Stub implementation - in production, this would call task service
        if task_type == "reminder":
            if due_date:
                response_text = f"Reminder set for {due_date}: {task_text}, Sir."
            else:
                response_text = f"Reminder set: {task_text}, Sir."
        
        elif task_type == "todo":
            response_text = f"Todo added: {task_text}, Sir."
        
        elif task_type == "schedule":
            if due_date:
                response_text = f"Scheduled for {due_date}: {task_text}, Sir."
            else:
                response_text = f"Task scheduled: {task_text}, Sir."
        
        else:
            response_text = f"Task created: {task_text}, Sir."
        
        action_spec = {
            "intent": "task_management",
            "task_type": task_type,
            "text": task_text,
            "due_date": due_date,
            "priority": priority,
            "requires_permission": False,
        }
        
        return HandlerResponse(
            text=response_text,
            action_spec=action_spec,
            metadata={
                "intent": "task_management",
                "task_type": task_type,
                "emotion": "helpful",
            }
        )


class ConversationHandler:
    """Handler for general conversation intents (Q&A, chit-chat, general queries)."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
    
    async def handle(
        self,
        text: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> HandlerResponse:
        """
        Handle conversation intent.
        
        Note: This handler is typically a pass-through to LLM.
        This stub provides fallback responses when LLM is unavailable.
        
        Args:
            text: Original user utterance
            parameters: Extracted parameters (usually empty for conversation)
            context: Conversation context
            
        Returns:
            HandlerResponse with normalized output
        """
        text_lower = text.lower()
        
        # Stub fallback responses - in production, this would call LLM
        if any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
            response_text = "Hello, Sir. How may I assist you today?"
        
        elif any(word in text_lower for word in ["how are you", "how do you feel"]):
            response_text = "I'm functioning perfectly, Sir. Thank you for asking."
        
        elif any(word in text_lower for word in ["what time", "what's the time", "time"]):
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            response_text = f"The current time is {current_time}, Sir."
        
        elif any(word in text_lower for word in ["what date", "what's the date", "date"]):
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            response_text = f"Today is {current_date}, Sir."
        
        elif any(word in text_lower for word in ["thank", "thanks"]):
            response_text = "You're welcome, Sir. Always at your service."
        
        elif any(word in text_lower for word in ["goodbye", "bye", "see you"]):
            response_text = "Goodbye, Sir. Have a pleasant day."
        
        else:
            # Generic fallback - in production, this should never be reached if LLM is working
            response_text = f"I understand you said: '{text}'. I'm processing that now, Sir."
        
        # Conversation intents don't typically generate action specs
        return HandlerResponse(
            text=response_text,
            action_spec=None,
            metadata={
                "intent": "conversation",
                "emotion": "friendly",
                "requires_llm": True,  # Indicates this should use LLM in production
            }
        )


# Handler registry
_HANDLERS = {
    "system_control": SystemControlHandler,
    "search": SearchHandler,
    "task_management": TaskManagementHandler,
    "conversation": ConversationHandler,
}


def get_handler(intent: str, settings: Optional[Settings] = None) -> Optional[Any]:
    """
    Get the appropriate handler for an intent.
    
    Args:
        intent: Intent label
        settings: Optional settings instance
        
    Returns:
        Handler instance or None if intent not found
    """
    handler_class = _HANDLERS.get(intent.lower())
    if handler_class:
        return handler_class(settings)
    return None


async def handle_intent(
    intent: str,
    text: str,
    parameters: Dict[str, Any],
    context: Dict[str, Any],
    settings: Optional[Settings] = None
) -> HandlerResponse:
    """
    Convenience function to handle an intent with the appropriate handler.
    
    Args:
        intent: Intent label
        text: Original user utterance
        parameters: Extracted parameters
        context: Conversation context
        settings: Optional settings instance
        
    Returns:
        HandlerResponse with normalized output
        
    Raises:
        ValueError: If intent has no handler
    """
    handler = get_handler(intent, settings)
    if handler is None:
        raise ValueError(f"No handler found for intent: {intent}")
    
    return await handler.handle(text, parameters, context)

