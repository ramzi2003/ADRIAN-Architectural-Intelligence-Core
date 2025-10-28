"""
Conversation Manager for ADRIAN IO Service.
Handles conversation state, context awareness, and continuous listening.
"""
import asyncio
import time
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from collections import deque

from shared.logging_config import get_logger
from .stt_utils import STTProcessor, AudioQuality, STTError, speech_to_text
from .vad_utils import VADState

logger = get_logger("io-service.conversation")


class ConversationState(Enum):
    """Conversation states for context awareness."""
    IDLE = "idle"  # Waiting for "ADRIAN" hotword
    LISTENING = "listening"  # After hotword, actively listening
    PROCESSING = "processing"  # STT + NLP processing
    RESPONDING = "responding"  # TTS output
    CONTINUOUS = "continuous"  # In conversation (no need to repeat "ADRIAN")


@dataclass
class ConversationContext:
    """Context information for conversation continuity."""
    user_id: str
    conversation_id: str
    start_time: float
    last_activity: float
    turn_count: int
    last_transcription: str
    conversation_history: List[Dict[str, str]]  # [{"role": "user/assistant", "text": "..."}]
    current_topic: Optional[str] = None
    confidence_history: List[float] = None
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = []


class ConversationManager:
    """
    Manages conversation flow and context awareness.
    Handles continuous listening and conversation state transitions.
    """
    
    def __init__(self, 
                 conversation_timeout: float = 30.0,  # 30 seconds of silence = back to idle
                 max_context_turns: int = 5,  # Remember last 5 exchanges
                 confidence_threshold: float = 0.7,
                 stt_processor: Optional[STTProcessor] = None):
        """
        Initialize conversation manager.
        
        Args:
            conversation_timeout: Seconds of silence before returning to idle
            max_context_turns: Maximum number of conversation turns to remember
            confidence_threshold: Minimum confidence for accepting transcription
            stt_processor: STTProcessor instance to use for transcription
        """
        self.conversation_timeout = conversation_timeout
        self.max_context_turns = max_context_turns
        self.confidence_threshold = confidence_threshold
        self.stt_processor = stt_processor
        
        # State management
        self.current_state = ConversationState.IDLE
        self.current_context: Optional[ConversationContext] = None
        
        # Audio buffering
        self.audio_buffer: deque = deque(maxlen=int(16000 * 10))  # 10 seconds of audio
        self.is_buffering = False
        self.buffer_start_time = 0.0
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_transcription: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Performance tracking
        self.conversation_stats = {
            "total_conversations": 0,
            "total_turns": 0,
            "avg_confidence": 0.0,
            "error_counts": {error.value: 0 for error in STTError}
        }
        
        logger.info(f"Conversation Manager initialized (timeout: {conversation_timeout}s, max_turns: {max_context_turns})")
    
    def start_conversation(self, user_id: str = "default_user") -> str:
        """
        Start a new conversation after hotword detection.
        
        Args:
            user_id: User identifier
            
        Returns:
            Conversation ID
        """
        conversation_id = f"conv_{int(time.time() * 1000)}"
        
        self.current_context = ConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            start_time=time.time(),
            last_activity=time.time(),
            turn_count=0,
            last_transcription="",
            conversation_history=[]
        )
        
        self.current_state = ConversationState.LISTENING
        self.is_buffering = True
        self.buffer_start_time = time.time()
        
        self.conversation_stats["total_conversations"] += 1
        
        logger.info(f"🎯 Conversation started: {conversation_id} (user: {user_id})")
        
        if self.on_state_change:
            self.on_state_change(self.current_state, self.current_context)
        
        return conversation_id
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """
        Add audio chunk to conversation buffer.
        
        Args:
            audio_data: Audio data as numpy array
        """
        if self.current_state in [ConversationState.LISTENING, ConversationState.CONTINUOUS]:
            self.audio_buffer.extend(audio_data)
            self.current_context.last_activity = time.time()
    
    async def process_audio_buffer(self) -> Tuple[str, float, bool]:
        """
        Process accumulated audio buffer for transcription.
        
        Returns:
            Tuple of (transcription, confidence, is_valid)
        """
        if not self.is_buffering or len(self.audio_buffer) == 0:
            return "", 0.0, False
        
        # Convert buffer to numpy array
        audio_data = np.array(list(self.audio_buffer), dtype=np.int16)
        duration = len(audio_data) / 16000  # 16kHz sample rate
        
        logger.info(f"📊 Audio buffer: {len(audio_data)} samples ({duration:.2f}s)")
        
        # Check if we have enough audio (at least 0.5 seconds)
        if len(audio_data) < 8000:  # Less than 0.5 seconds at 16kHz
            logger.warning(f"Audio too short: {duration:.2f}s")
            return "", 0.0, False
        
        # Transcribe audio
        if self.stt_processor:
            transcription, confidence, audio_quality, error_type = await self.stt_processor.transcribe_audio(audio_data)
        else:
            # Fallback to global function
            transcription, confidence, audio_quality, error_type = await speech_to_text(audio_data)
        
        # Update context
        if self.current_context:
            self.current_context.last_transcription = transcription
            self.current_context.confidence_history.append(confidence)
            
            # Keep only recent confidence scores
            if len(self.current_context.confidence_history) > 10:
                self.current_context.confidence_history = self.current_context.confidence_history[-10:]
        
        # Handle errors
        if error_type:
            self.conversation_stats["error_counts"][error_type.value] += 1
            logger.warning(f"STT Error: {error_type.value} (confidence: {confidence:.2f})")
            
            if self.on_error:
                await self.on_error(error_type, audio_quality, transcription)
            
            return transcription, confidence, False
        
        # Check confidence threshold
        is_valid = confidence >= self.confidence_threshold
        
        if is_valid:
            # Add to conversation history
            if self.current_context:
                self.current_context.conversation_history.append({
                    "role": "user",
                    "text": transcription,
                    "timestamp": time.time(),
                    "confidence": confidence
                })
                
                # Keep only recent history
                if len(self.current_context.conversation_history) > self.max_context_turns * 2:
                    self.current_context.conversation_history = self.current_context.conversation_history[-(self.max_context_turns * 2):]
                
                self.current_context.turn_count += 1
                self.conversation_stats["total_turns"] += 1
            
            # Update state
            self.current_state = ConversationState.CONTINUOUS
            
            logger.info(f"🎤 Transcription: '{transcription}' (confidence: {confidence:.2f})")
            
            if self.on_transcription:
                await self.on_transcription(transcription, confidence, self.current_context)
        
        return transcription, confidence, is_valid
    
    def should_continue_listening(self) -> bool:
        """
        Check if we should continue listening based on timeout and activity.
        
        Returns:
            True if should continue, False if should return to idle
        """
        if not self.current_context:
            return False
        
        time_since_activity = time.time() - self.current_context.last_activity
        return time_since_activity < self.conversation_timeout
    
    def end_conversation(self):
        """End current conversation and return to idle state."""
        if self.current_context:
            duration = time.time() - self.current_context.start_time
            logger.info(f"🔚 Conversation ended: {self.current_context.conversation_id} (duration: {duration:.1f}s, turns: {self.current_context.turn_count})")
        
        self.current_context = None
        self.current_state = ConversationState.IDLE
        self.is_buffering = False
        self.audio_buffer.clear()
        
        if self.on_state_change:
            self.on_state_change(self.current_state, None)
    
    def get_conversation_context(self) -> Optional[ConversationContext]:
        """Get current conversation context."""
        return self.current_context
    
    def get_current_state(self) -> ConversationState:
        """Get current conversation state."""
        return self.current_state
    
    def is_in_conversation(self) -> bool:
        """Check if currently in an active conversation."""
        return self.current_state != ConversationState.IDLE
    
    def get_conversation_summary(self) -> Dict:
        """Get conversation summary for context awareness."""
        if not self.current_context:
            return {}
        
        return {
            "conversation_id": self.current_context.conversation_id,
            "user_id": self.current_context.user_id,
            "turn_count": self.current_context.turn_count,
            "duration": time.time() - self.current_context.start_time,
            "last_transcription": self.current_context.last_transcription,
            "recent_history": self.current_context.conversation_history[-3:],  # Last 3 exchanges
            "avg_confidence": np.mean(self.current_context.confidence_history) if self.current_context.confidence_history else 0.0,
            "current_state": self.current_state.value
        }
    
    def get_performance_stats(self) -> Dict:
        """Get conversation performance statistics."""
        stats = self.conversation_stats.copy()
        
        if self.current_context and self.current_context.confidence_history:
            stats["current_avg_confidence"] = np.mean(self.current_context.confidence_history)
        
        return stats


# Global conversation manager instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get global conversation manager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


def test_conversation_manager():
    """Test conversation manager functionality."""
    print("🧪 Testing Conversation Manager...")
    
    manager = ConversationManager()
    
    # Test conversation start
    conv_id = manager.start_conversation("test_user")
    print(f"✅ Conversation started: {conv_id}")
    
    # Test state
    print(f"📊 Current state: {manager.get_current_state().value}")
    print(f"📊 In conversation: {manager.is_in_conversation()}")
    
    # Test context
    context = manager.get_conversation_context()
    print(f"📊 Context: {context.conversation_id if context else 'None'}")
    
    # Test conversation end
    manager.end_conversation()
    print(f"📊 After end - State: {manager.get_current_state().value}")
    print(f"📊 After end - In conversation: {manager.is_in_conversation()}")
    
    # Test performance stats
    stats = manager.get_performance_stats()
    print(f"📈 Performance stats: {stats}")


if __name__ == "__main__":
    test_conversation_manager()
