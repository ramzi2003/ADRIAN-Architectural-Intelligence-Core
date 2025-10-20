"""
Voice Activity Detection utilities for ADRIAN IO Service.
Uses WebRTC VAD for real-time speech detection.
"""
import webrtcvad
import numpy as np
from typing import Optional, Tuple, List
from enum import Enum
import time
from collections import deque

from shared.logging_config import get_logger

logger = get_logger("io-service.vad")

# VAD Configuration
SAMPLE_RATE = 16000  # Hz - Must be 16kHz for WebRTC VAD
CHUNK_DURATION_MS = 30  # ms - 30ms chunks for better accuracy
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 480 samples at 16kHz

# State management timing (in seconds)
MINIMUM_SPEECH_DURATION = 0.2  # 200ms minimum speech
SILENCE_TIMEOUT = 0.5  # 500ms silence before "stop speaking"
VAD_DECISION_BUFFER = 3  # Number of chunks to buffer for smooth transitions


class VADState(Enum):
    """Voice activity detection states."""
    SILENCE = "silence"
    SPEECH = "speech"
    TRANSITION = "transition"  # Brief state during transitions


class VADStateManager:
    """
    Manages voice activity detection state with buffering and timing.
    """
    
    def __init__(
        self,
        aggressiveness: int = 1,
        minimum_speech_duration: float = MINIMUM_SPEECH_DURATION,
        silence_timeout: float = SILENCE_TIMEOUT,
        buffer_size: int = VAD_DECISION_BUFFER
    ):
        """
        Initialize VAD state manager.
        
        Args:
            aggressiveness: VAD sensitivity (0=most aggressive, 3=most conservative)
            minimum_speech_duration: Minimum speech duration to consider valid
            silence_timeout: Silence duration to trigger "stop speaking"
            buffer_size: Number of chunks to buffer for decision smoothing
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.minimum_speech_duration = minimum_speech_duration
        self.silence_timeout = silence_timeout
        self.buffer_size = buffer_size
        
        # State tracking
        self.current_state = VADState.SILENCE
        self.last_speech_time = 0.0
        self.last_silence_time = 0.0
        self.speech_start_time = 0.0
        self.speech_end_time = 0.0
        
        # Decision buffer for smoothing
        self.decision_buffer: deque = deque(maxlen=buffer_size)
        self.transition_threshold = buffer_size // 2 + 1
        
        logger.info(f"VAD State Manager initialized: aggressiveness={aggressiveness}, "
                   f"min_speech={minimum_speech_duration}s, silence_timeout={silence_timeout}s")
    
    def _smooth_decision(self, current_decision: bool) -> bool:
        """
        Apply smoothing to VAD decisions using a buffer.
        
        Args:
            current_decision: Raw VAD decision for current chunk
            
        Returns:
            Smoothed decision
        """
        self.decision_buffer.append(current_decision)
        
        # Need enough samples to make a decision
        if len(self.decision_buffer) < self.buffer_size:
            return current_decision
        
        # Count true decisions in buffer
        true_count = sum(self.decision_buffer)
        
        # Use majority vote for smoothing
        return true_count >= self.transition_threshold
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, VADState]:
        """
        Process audio chunk and return speech detection result with state.
        
        Args:
            audio_chunk: Audio data as numpy array (int16, mono, 16kHz)
            
        Returns:
            Tuple of (is_speech, current_state)
        """
        if len(audio_chunk) != CHUNK_SIZE:
            logger.warning(f"Audio chunk size {len(audio_chunk)} != expected {CHUNK_SIZE}")
            return False, self.current_state
        
        current_time = time.time()
        
        try:
            # Convert numpy array to bytes for webrtcvad
            audio_bytes = audio_chunk.astype(np.int16).tobytes()
            
            # Get raw VAD decision
            raw_decision = self.vad.is_speech(audio_bytes, SAMPLE_RATE)
            
            # Apply smoothing
            smoothed_decision = self._smooth_decision(raw_decision)
            
            # Update state based on smoothed decision
            self._update_state(smoothed_decision, current_time)
            
            return smoothed_decision, self.current_state
            
        except Exception as e:
            logger.error(f"Error processing VAD chunk: {e}")
            return False, self.current_state
    
    def _update_state(self, is_speech: bool, current_time: float):
        """Update internal state based on speech detection."""
        
        if is_speech:
            self.last_speech_time = current_time
            
            # Transition to speech state
            if self.current_state == VADState.SILENCE:
                self.speech_start_time = current_time
                self.current_state = VADState.TRANSITION
                logger.debug("Transitioning to speech state")
            
            # Confirm speech if we've been speaking long enough
            elif (self.current_state == VADState.TRANSITION and 
                  current_time - self.speech_start_time >= self.minimum_speech_duration):
                self.current_state = VADState.SPEECH
                logger.debug("Confirmed speech state")
                
        else:
            self.last_silence_time = current_time
            
            # Transition to silence
            if self.current_state == VADState.SPEECH:
                self.current_state = VADState.TRANSITION
                logger.debug("Transitioning to silence state")
            
            # Confirm silence if timeout reached
            elif (self.current_state == VADState.TRANSITION and 
                  current_time - self.last_speech_time >= self.silence_timeout):
                self.speech_end_time = current_time
                self.current_state = VADState.SILENCE
                logger.debug("Confirmed silence state")
    
    def is_speaking(self) -> bool:
        """Check if currently in speech state."""
        return self.current_state == VADState.SPEECH
    
    def has_speech_started(self) -> bool:
        """Check if we just transitioned from silence to speech."""
        return (self.current_state == VADState.SPEECH and 
                self.speech_start_time > 0 and 
                time.time() - self.speech_start_time < 0.1)  # Within 100ms of start
    
    def has_speech_ended(self) -> bool:
        """Check if we just transitioned from speech to silence."""
        return (self.current_state == VADState.SILENCE and 
                self.speech_end_time > 0 and 
                time.time() - self.speech_end_time < 0.1)  # Within 100ms of end
    
    def get_speech_duration(self) -> float:
        """Get duration of current or last speech segment."""
        if self.current_state == VADState.SPEECH:
            return time.time() - self.speech_start_time
        elif self.speech_end_time > 0:
            return self.speech_end_time - self.speech_start_time
        return 0.0
    
    def reset(self):
        """Reset state manager to initial state."""
        self.current_state = VADState.SILENCE
        self.last_speech_time = 0.0
        self.last_silence_time = 0.0
        self.speech_start_time = 0.0
        self.speech_end_time = 0.0
        self.decision_buffer.clear()
        logger.debug("VAD state manager reset")


def detect_speech(
    audio_chunk: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    aggressiveness: int = 1
) -> bool:
    """
    Simple speech detection function for single audio chunks.
    
    Args:
        audio_chunk: Audio data as numpy array (int16, mono)
        sample_rate: Sample rate (must be 16kHz for WebRTC VAD)
        aggressiveness: VAD sensitivity (0-3)
        
    Returns:
        True if speech detected, False otherwise
    """
    if sample_rate != SAMPLE_RATE:
        logger.error(f"Sample rate {sample_rate} not supported. Must be {SAMPLE_RATE}Hz")
        return False
    
    if len(audio_chunk) not in [160, 320, 480]:  # 10ms, 20ms, 30ms at 16kHz
        logger.error(f"Chunk size {len(audio_chunk)} not supported by WebRTC VAD")
        return False
    
    try:
        vad = webrtcvad.Vad(aggressiveness)
        audio_bytes = audio_chunk.astype(np.int16).tobytes()
        return vad.is_speech(audio_bytes, sample_rate)
    
    except Exception as e:
        logger.error(f"Error in speech detection: {e}")
        return False


def prepare_audio_for_vad(audio_data: np.ndarray, target_length: int = CHUNK_SIZE) -> List[np.ndarray]:
    """
    Prepare audio data for VAD processing by chunking to proper size.
    
    Args:
        audio_data: Input audio array
        target_length: Target chunk size (default 480 for 30ms at 16kHz)
        
    Returns:
        List of audio chunks ready for VAD processing
    """
    chunks = []
    
    # Pad or truncate to multiple of target_length
    remainder = len(audio_data) % target_length
    if remainder > 0:
        padding = target_length - remainder
        audio_data = np.pad(audio_data, (0, padding), mode='constant')
    
    # Split into chunks
    for i in range(0, len(audio_data), target_length):
        chunk = audio_data[i:i + target_length]
        if len(chunk) == target_length:
            chunks.append(chunk)
    
    return chunks


# Global VAD state manager instance
_vad_manager: Optional[VADStateManager] = None


def get_vad_manager() -> VADStateManager:
    """Get or create global VAD state manager."""
    global _vad_manager
    if _vad_manager is None:
        _vad_manager = VADStateManager()
    return _vad_manager


def initialize_vad(aggressiveness: int = 1) -> VADStateManager:
    """
    Initialize VAD system with specified parameters.
    
    Args:
        aggressiveness: VAD sensitivity level (0-3)
        
    Returns:
        Initialized VAD state manager
    """
    global _vad_manager
    _vad_manager = VADStateManager(aggressiveness=aggressiveness)
    logger.info(f"VAD initialized with aggressiveness={aggressiveness}")
    return _vad_manager
