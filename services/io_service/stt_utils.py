"""
Speech-to-Text utilities for ADRIAN IO Service.
Uses OpenAI Whisper for high-quality speech recognition with intelligent error handling.
"""
import whisper
import numpy as np
import asyncio
import tempfile
import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import time
from enum import Enum

from shared.logging_config import get_logger

logger = get_logger("io-service.stt")

# STT Configuration
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for accepting transcription
AUDIO_QUALITY_THRESHOLD = 0.3  # Minimum audio quality for processing
MAX_AUDIO_DURATION = 30  # Maximum audio duration in seconds
SAMPLE_RATE = 16000  # Hz - Must match audio pipeline


class AudioQuality(Enum):
    """Audio quality classifications."""
    CLEAR = "clear"
    BACKGROUND_NOISE = "background_noise"
    MUSIC = "music"
    MULTIPLE_SPEAKERS = "multiple_speakers"
    UNCLEAR = "unclear"
    TOO_SHORT = "too_short"
    TOO_QUIET = "too_quiet"


class STTError(Enum):
    """STT error types for intelligent handling."""
    LOW_CONFIDENCE = "low_confidence"
    POOR_AUDIO_QUALITY = "poor_audio_quality"
    BACKGROUND_NOISE = "background_noise"
    MUSIC_DETECTED = "music_detected"
    MULTIPLE_SPEAKERS = "multiple_speakers"
    TOO_SHORT = "too_short"
    NO_SPEECH = "no_speech"
    PROCESSING_ERROR = "processing_error"


class STTProcessor:
    """
    Speech-to-Text processor using OpenAI Whisper.
    Handles audio quality detection and intelligent error classification.
    """
    
    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        """
        Initialize STT processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model: Optional[whisper.Whisper] = None
        self.is_loaded = False
        self.load_time = 0.0
        
        # Performance tracking
        self.processing_times = []
        self.confidence_scores = []
        self.error_counts = {error_type.value: 0 for error_type in STTError}
        
        logger.info(f"STT Processor initialized with model size: {model_size}")
    
    def load_model(self) -> bool:
        """
        Load Whisper model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            start_time = time.time()
            
            # Load the model
            self.model = whisper.load_model(self.model_size)
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"✅ Whisper model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def analyze_audio_quality(self, audio_data: np.ndarray) -> Tuple[AudioQuality, float]:
        """
        Analyze audio quality and classify potential issues.
        
        Args:
            audio_data: Audio data as numpy array (int16, mono, 16kHz)
            
        Returns:
            Tuple of (quality_classification, quality_score)
        """
        if len(audio_data) == 0:
            return AudioQuality.TOO_SHORT, 0.0
        
        # Convert to float for analysis
        audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        
        # Basic audio quality metrics
        duration = len(audio_float) / SAMPLE_RATE
        rms_energy = np.sqrt(np.mean(audio_float ** 2))
        max_amplitude = np.max(np.abs(audio_float))
        
        # Check for too short audio
        if duration < 0.5:  # Less than 500ms
            return AudioQuality.TOO_SHORT, 0.0
        
        # Check for too quiet audio
        if rms_energy < 0.01:  # Very low energy
            return AudioQuality.TOO_QUIET, rms_energy
        
        # Check for clipping (distortion)
        clipping_ratio = np.sum(np.abs(audio_float) > 0.95) / len(audio_float)
        if clipping_ratio > 0.1:  # More than 10% clipped
            return AudioQuality.UNCLEAR, 1.0 - clipping_ratio
        
        # Check for background noise (high frequency content)
        # Simple frequency analysis
        fft = np.fft.fft(audio_float)
        freqs = np.fft.fftfreq(len(audio_float), 1/SAMPLE_RATE)
        
        # High frequency energy (potential background noise)
        high_freq_mask = np.abs(freqs) > 4000  # Above 4kHz
        high_freq_energy = np.sum(np.abs(fft[high_freq_mask]))
        total_energy = np.sum(np.abs(fft))
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
            if high_freq_ratio > 0.3:  # High frequency content suggests noise
                return AudioQuality.BACKGROUND_NOISE, 1.0 - high_freq_ratio
        
        # Check for music (rhythmic patterns)
        # Simple rhythm detection using autocorrelation
        autocorr = np.correlate(audio_float, audio_float, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Look for periodic patterns (music typically has rhythm)
        if len(autocorr) > 100:
            # Check for peaks in autocorrelation (rhythmic patterns)
            peaks = []
            for i in range(50, min(200, len(autocorr))):  # 50-200 sample range
                if (autocorr[i] > autocorr[i-1] and 
                    autocorr[i] > autocorr[i+1] and 
                    autocorr[i] > np.mean(autocorr) * 1.2):
                    peaks.append(i)
            
            if len(peaks) > 3:  # Multiple rhythmic peaks suggest music
                return AudioQuality.MUSIC, 0.5
        
        # Calculate overall quality score
        quality_score = min(1.0, rms_energy * 10)  # Scale RMS energy
        
        if quality_score > AUDIO_QUALITY_THRESHOLD:
            return AudioQuality.CLEAR, quality_score
        else:
            return AudioQuality.UNCLEAR, quality_score
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Tuple[str, float, AudioQuality, Optional[STTError]]:
        """
        Transcribe audio data using Whisper.
        
        Args:
            audio_data: Audio data as numpy array (int16, mono, 16kHz)
            
        Returns:
            Tuple of (transcription, confidence, audio_quality, error_type)
        """
        if not self.is_loaded:
            if not self.load_model():
                return "", 0.0, AudioQuality.UNCLEAR, STTError.PROCESSING_ERROR
        
        try:
            start_time = time.time()
            
            # Analyze audio quality first
            audio_quality, quality_score = self.analyze_audio_quality(audio_data)
            
            # Handle poor quality audio
            if audio_quality in [AudioQuality.TOO_SHORT, AudioQuality.TOO_QUIET]:
                error_type = STTError.TOO_SHORT if audio_quality == AudioQuality.TOO_SHORT else STTError.POOR_AUDIO_QUALITY
                self.error_counts[error_type.value] += 1
                return "", 0.0, audio_quality, error_type
            
            # Convert audio to the format Whisper expects
            audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            
            # Save to temporary file for Whisper processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write audio data to temporary file
                import soundfile as sf
                sf.write(temp_path, audio_float, SAMPLE_RATE)
            
            try:
                # Transcribe using Whisper
                result = self.model.transcribe(
                    temp_path,
                    language="en",  # English for now
                    fp16=False,  # Use fp32 for better compatibility
                    verbose=False
                )
                
                # Extract transcription and confidence
                transcription = result["text"].strip()
                confidence = 1.0  # Whisper doesn't provide confidence scores directly
                
                # Estimate confidence based on audio quality and transcription length
                if len(transcription) == 0:
                    confidence = 0.0
                    error_type = STTError.NO_SPEECH
                elif quality_score < AUDIO_QUALITY_THRESHOLD:
                    confidence = quality_score
                    error_type = STTError.POOR_AUDIO_QUALITY
                elif audio_quality == AudioQuality.BACKGROUND_NOISE:
                    confidence = 0.6
                    error_type = STTError.BACKGROUND_NOISE
                elif audio_quality == AudioQuality.MUSIC:
                    confidence = 0.4
                    error_type = STTError.MUSIC_DETECTED
                else:
                    # Good quality audio
                    confidence = min(0.9, quality_score + 0.3)
                    error_type = None
                
                # Check confidence threshold
                if confidence < CONFIDENCE_THRESHOLD:
                    if error_type is None:
                        error_type = STTError.LOW_CONFIDENCE
                    self.error_counts[error_type.value] += 1
                
                # Track performance metrics
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.confidence_scores.append(confidence)
                
                logger.info(f"STT Result: '{transcription}' (confidence: {confidence:.2f}, quality: {audio_quality.value}, time: {processing_time:.2f}s)")
                
                return transcription, confidence, audio_quality, error_type
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"STT processing error: {e}")
            self.error_counts[STTError.PROCESSING_ERROR.value] += 1
            return "", 0.0, AudioQuality.UNCLEAR, STTError.PROCESSING_ERROR
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {"status": "no_data"}
        
        return {
            "model_size": self.model_size,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "total_transcriptions": len(self.processing_times),
            "avg_processing_time": np.mean(self.processing_times),
            "avg_confidence": np.mean(self.confidence_scores) if self.confidence_scores else 0.0,
            "error_counts": self.error_counts,
            "recent_processing_times": self.processing_times[-10:],  # Last 10
            "recent_confidence_scores": self.confidence_scores[-10:]  # Last 10
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.processing_times.clear()
        self.confidence_scores.clear()
        self.error_counts = {error_type.value: 0 for error_type in STTError}
        logger.info("STT performance statistics reset")


# Global STT processor instance
_stt_processor: Optional[STTProcessor] = None


def get_stt_processor() -> STTProcessor:
    """Get global STT processor instance."""
    global _stt_processor
    if _stt_processor is None:
        _stt_processor = STTProcessor()
    return _stt_processor


async def speech_to_text(audio_data: np.ndarray) -> Tuple[str, float, AudioQuality, Optional[STTError]]:
    """
    Convert speech audio to text.
    
    Args:
        audio_data: Audio data as numpy array (int16, mono, 16kHz)
        
    Returns:
        Tuple of (transcription, confidence, audio_quality, error_type)
    """
    processor = get_stt_processor()
    return await processor.transcribe_audio(audio_data)


async def test_stt_processor():
    """Test STT processor functionality."""
    print("🧪 Testing STT Processor...")
    
    processor = STTProcessor()
    
    # Test model loading
    if processor.load_model():
        print("✅ Model loaded successfully")
        
        # Test with dummy audio data
        dummy_audio = np.random.randint(-1000, 1000, size=SAMPLE_RATE * 2, dtype=np.int16)  # 2 seconds
        
        # Test audio quality analysis
        quality, score = processor.analyze_audio_quality(dummy_audio)
        print(f"📊 Audio quality: {quality.value} (score: {score:.2f})")
        
        # Test transcription (will likely fail with dummy data, but tests the pipeline)
        result = await processor.transcribe_audio(dummy_audio)
        print(f"🎤 Transcription result: {result}")
        
        # Show performance stats
        stats = processor.get_performance_stats()
        print(f"📈 Performance stats: {stats}")
        
    else:
        print("❌ Failed to load model")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_stt_processor())
