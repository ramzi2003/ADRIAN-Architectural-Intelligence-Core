"""
Personality-driven error handling for ADRIAN.
Provides witty, context-aware responses for STT failures and audio quality issues.
"""
import random
from typing import Dict, List, Optional
from enum import Enum

from shared.logging_config import get_logger
from .stt_utils import AudioQuality, STTError

logger = get_logger("io-service.personality")


class ErrorSeverity(Enum):
    """Error severity levels for appropriate responses."""
    MILD = "mild"  # Low confidence, minor issues
    MODERATE = "moderate"  # Background noise, unclear speech
    SEVERE = "severe"  # Music, multiple speakers, processing errors


class PersonalityErrorHandler:
    """
    Generates personality-driven error responses for STT failures.
    Maintains ADRIAN's witty, helpful character while providing useful feedback.
    """
    
    def __init__(self):
        """Initialize personality error handler."""
        self.error_responses = self._initialize_error_responses()
        self.context_responses = self._initialize_context_responses()
        self.retry_attempts = {}  # Track retry attempts per conversation
        
        logger.info("Personality Error Handler initialized")
    
    def _initialize_error_responses(self) -> Dict[STTError, List[str]]:
        """Initialize error-specific response templates."""
        return {
            STTError.LOW_CONFIDENCE: [
                "I beg your pardon, Sir, but I'm having trouble making out your words. Could you speak a bit more clearly?",
                "My apologies, Sir, but your words seem to be getting lost in transmission. Perhaps you could repeat that?",
                "I'm afraid I'm not catching everything you're saying, Sir. Could you try again?",
                "Your voice seems a bit unclear to me, Sir. Would you mind speaking up a little?",
                "I'm having difficulty understanding you, Sir. Could you rephrase that for me?"
            ],
            
            STTError.POOR_AUDIO_QUALITY: [
                "I'm detecting some audio interference, Sir. The signal quality seems rather poor at the moment.",
                "There appears to be some technical difficulty with the audio input, Sir. Could you try again?",
                "I'm experiencing some audio quality issues, Sir. The sound seems distorted.",
                "The audio signal is rather weak, Sir. Perhaps you could speak closer to the microphone?",
                "I'm having trouble with the audio quality, Sir. There seems to be some interference."
            ],
            
            STTError.BACKGROUND_NOISE: [
                "I can hear you, Sir, but there's quite a bit of background noise interfering with our conversation.",
                "There's some distracting noise in the background, Sir. Could you move to a quieter location?",
                "I'm picking up quite a bit of background chatter, Sir. It's making it difficult to focus on your words.",
                "The ambient noise level is rather high, Sir. Perhaps we could find a quieter spot for our conversation?",
                "I can hear you, but there's significant background noise, Sir. Could you reduce the ambient sound?"
            ],
            
            STTError.MUSIC_DETECTED: [
                "I detect music playing in the background, Sir. While I appreciate your taste, it's making it difficult to understand you.",
                "There seems to be music playing, Sir. Could you pause it so we can have a proper conversation?",
                "I can hear some music in the background, Sir. It's rather distracting from our discussion.",
                "The musical accompaniment is lovely, Sir, but it's interfering with our communication.",
                "I'm detecting background music, Sir. Perhaps we could continue our conversation in a quieter environment?"
            ],
            
            STTError.MULTIPLE_SPEAKERS: [
                "I can hear multiple voices, Sir. Could you ensure we're having a private conversation?",
                "There seem to be several people speaking, Sir. I'm having trouble focusing on your voice specifically.",
                "I detect multiple speakers, Sir. Could you speak when others aren't talking?",
                "There's quite a bit of conversation happening, Sir. Could you speak when it's quieter?",
                "I can hear several voices, Sir. It's challenging to distinguish your words from the others."
            ],
            
            STTError.TOO_SHORT: [
                "That was rather brief, Sir. Could you provide a bit more detail?",
                "I didn't quite catch that, Sir. Could you elaborate a bit more?",
                "That seemed like a very short message, Sir. Could you say a bit more?",
                "I'm not sure I got the full message, Sir. Could you expand on that?",
                "That was quite brief, Sir. Could you provide more context?"
            ],
            
            STTError.NO_SPEECH: [
                "I'm not detecting any speech, Sir. Are you there?",
                "I don't hear anything, Sir. Could you speak up?",
                "The microphone seems quiet, Sir. Could you try speaking?",
                "I'm not picking up any audio, Sir. Is everything working properly?",
                "I don't detect any speech, Sir. Could you check your microphone?"
            ],
            
            STTError.PROCESSING_ERROR: [
                "I'm experiencing some technical difficulties, Sir. Let me try to resolve this.",
                "There seems to be a processing error, Sir. I'll attempt to fix this.",
                "I'm having some technical trouble, Sir. Please bear with me while I sort this out.",
                "There's a technical issue on my end, Sir. I'll work to resolve it.",
                "I'm encountering a processing error, Sir. Let me try a different approach."
            ]
        }
    
    def _initialize_context_responses(self) -> Dict[str, List[str]]:
        """Initialize context-aware response templates."""
        return {
            "first_attempt": [
                "Let me try that again, Sir.",
                "One more time, Sir.",
                "Let me give that another go, Sir.",
                "I'll try once more, Sir.",
                "Let me attempt that again, Sir."
            ],
            
            "retry_attempt": [
                "I'm still having trouble, Sir. Could you try speaking more slowly?",
                "I'm still not getting it clearly, Sir. Perhaps you could enunciate more?",
                "I'm still struggling with that, Sir. Could you try a different approach?",
                "I'm still having difficulty, Sir. Could you speak more distinctly?",
                "I'm still not catching it, Sir. Could you try rephrasing?"
            ],
            
            "conversation_context": [
                "In the context of our conversation, Sir, I'm having trouble with that part.",
                "Regarding what we were discussing, Sir, I'm not quite getting that.",
                "In relation to our topic, Sir, I'm having difficulty understanding.",
                "Building on our conversation, Sir, I'm not catching that clearly.",
                "Following up on our discussion, Sir, I'm having trouble with that."
            ],
            
            "encouragement": [
                "Don't worry, Sir, we'll get this sorted out.",
                "No problem, Sir, these things happen.",
                "It's quite alright, Sir, let's try again.",
                "Not to worry, Sir, I'm here to help.",
                "That's perfectly fine, Sir, let's continue."
            ]
        }
    
    def get_error_response(self, 
                          error_type: STTError, 
                          audio_quality: AudioQuality,
                          conversation_context: Optional[Dict] = None,
                          retry_count: int = 0) -> str:
        """
        Generate a personality-driven error response.
        
        Args:
            error_type: Type of STT error
            audio_quality: Audio quality classification
            conversation_context: Current conversation context
            retry_count: Number of retry attempts
            
        Returns:
            Personality-driven error response
        """
        # Determine error severity
        severity = self._determine_error_severity(error_type, audio_quality)
        
        # Get base response
        base_responses = self.error_responses.get(error_type, ["I'm having trouble understanding, Sir."])
        base_response = random.choice(base_responses)
        
        # Add context if available
        if conversation_context and conversation_context.get("turn_count", 0) > 0:
            context_responses = self.context_responses["conversation_context"]
            context_prefix = random.choice(context_responses)
            base_response = f"{context_prefix} {base_response.lower()}"
        
        # Add retry context
        if retry_count > 0:
            retry_responses = self.context_responses["retry_attempt"]
            retry_prefix = random.choice(retry_responses)
            base_response = f"{retry_prefix} {base_response.lower()}"
        
        # Add encouragement for severe errors
        if severity == ErrorSeverity.SEVERE:
            encouragement = random.choice(self.context_responses["encouragement"])
            base_response = f"{base_response} {encouragement}"
        
        # Add specific suggestions based on error type
        suggestion = self._get_error_suggestion(error_type, audio_quality)
        if suggestion:
            base_response = f"{base_response} {suggestion}"
        
        logger.info(f"Generated error response for {error_type.value}: {base_response}")
        return base_response
    
    def _determine_error_severity(self, error_type: STTError, audio_quality: AudioQuality) -> ErrorSeverity:
        """Determine error severity based on error type and audio quality."""
        if error_type in [STTError.PROCESSING_ERROR, STTError.MUSIC_DETECTED, STTError.MULTIPLE_SPEAKERS]:
            return ErrorSeverity.SEVERE
        elif error_type in [STTError.BACKGROUND_NOISE, STTError.POOR_AUDIO_QUALITY]:
            return ErrorSeverity.MODERATE
        else:
            return ErrorSeverity.MILD
    
    def _get_error_suggestion(self, error_type: STTError, audio_quality: AudioQuality) -> str:
        """Get specific suggestions based on error type."""
        suggestions = {
            STTError.BACKGROUND_NOISE: "Perhaps you could move to a quieter location, Sir?",
            STTError.MUSIC_DETECTED: "Could you pause the music temporarily, Sir?",
            STTError.MULTIPLE_SPEAKERS: "Could you ensure we're having a private conversation, Sir?",
            STTError.LOW_CONFIDENCE: "Could you speak more clearly and slowly, Sir?",
            STTError.PROCESSING_ERROR: "Let me try restarting the audio processing, Sir."
        }
        
        # Special-case suggestions based on measured audio quality
        if audio_quality == AudioQuality.TOO_QUIET:
            return "Could you speak a bit louder, Sir?"

        return suggestions.get(error_type, "")
    
    def get_retry_response(self, conversation_id: str) -> str:
        """
        Get response for retry attempts.
        
        Args:
            conversation_id: Current conversation ID
            
        Returns:
            Retry response
        """
        # Track retry attempts
        if conversation_id not in self.retry_attempts:
            self.retry_attempts[conversation_id] = 0
        
        self.retry_attempts[conversation_id] += 1
        retry_count = self.retry_attempts[conversation_id]
        
        if retry_count == 1:
            responses = self.context_responses["first_attempt"]
        else:
            responses = self.context_responses["retry_attempt"]
        
        return random.choice(responses)
    
    def clear_retry_attempts(self, conversation_id: str):
        """Clear retry attempts for a conversation."""
        if conversation_id in self.retry_attempts:
            del self.retry_attempts[conversation_id]
    
    def get_success_response(self, transcription: str, confidence: float) -> str:
        """
        Get response for successful transcription.
        
        Args:
            transcription: Transcribed text
            confidence: Confidence score
            
        Returns:
            Success response
        """
        if confidence > 0.9:
            return f"Perfect, Sir. I understood: '{transcription}'"
        elif confidence > 0.8:
            return f"Understood, Sir: '{transcription}'"
        else:
            return f"I believe you said: '{transcription}'"


# Global personality error handler instance
_personality_handler: Optional[PersonalityErrorHandler] = None


def get_personality_handler() -> PersonalityErrorHandler:
    """Get global personality error handler instance."""
    global _personality_handler
    if _personality_handler is None:
        _personality_handler = PersonalityErrorHandler()
    return _personality_handler


def test_personality_handler():
    """Test personality error handler functionality."""
    print("🧪 Testing Personality Error Handler...")
    
    handler = PersonalityErrorHandler()
    
    # Test different error types
    for error_type in STTError:
        response = handler.get_error_response(error_type, AudioQuality.CLEAR)
        print(f"📝 {error_type.value}: {response}")
    
    # Test retry responses
    print(f"\n🔄 Retry responses:")
    for i in range(3):
        response = handler.get_retry_response("test_conv")
        print(f"   Attempt {i+1}: {response}")
    
    # Test success responses
    print(f"\n✅ Success responses:")
    print(f"   High confidence: {handler.get_success_response('Hello ADRIAN', 0.95)}")
    print(f"   Medium confidence: {handler.get_success_response('Hello ADRIAN', 0.85)}")
    print(f"   Low confidence: {handler.get_success_response('Hello ADRIAN', 0.75)}")


if __name__ == "__main__":
    test_personality_handler()
