"""
Audio utilities for ADRIAN IO Service.
Handles microphone input and speaker output using sounddevice.
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Optional, List, Tuple, Callable
from pathlib import Path
import queue
import threading

from shared.logging_config import get_logger

# Import VAD utilities
try:
    from .vad_utils import VADStateManager, VADState, CHUNK_SIZE as VAD_CHUNK_SIZE
except ImportError:
    # Handle relative import issues
    VADStateManager = None
    VADState = None
    VAD_CHUNK_SIZE = 480

# Import Hotword utilities
try:
    from .hotword_utils import HotwordDetector, CHUNK_SIZE as HOTWORD_CHUNK_SIZE
except ImportError:
    HotwordDetector = None
    HOTWORD_CHUNK_SIZE = 512

logger = get_logger("io-service.audio")

# Audio configuration constants
SAMPLE_RATE = 16000  # Hz - Optimal for speech recognition (Whisper default)
CHANNELS = 1  # Mono audio for speech
DTYPE = 'int16'  # 16-bit PCM
CHUNK_SIZE = 480  # Samples per chunk (30ms at 16kHz for VAD compatibility)

# Global audio queue for streaming
audio_queue: queue.Queue = queue.Queue()


def list_audio_devices() -> List[dict]:
    """
    List all available audio input and output devices.
    Returns list of device info dictionaries.
    """
    devices = sd.query_devices()
    
    result = []
    for idx, device in enumerate(devices):
        result.append({
            'index': idx,
            'name': device['name'],
            'channels_in': device['max_input_channels'],
            'channels_out': device['max_output_channels'],
            'sample_rate': device['default_samplerate'],
            'is_input': device['max_input_channels'] > 0,
            'is_output': device['max_output_channels'] > 0
        })
    
    return result


def get_default_input_device() -> Optional[int]:
    """Get the default input device index."""
    try:
        return sd.default.device[0]
    except Exception as e:
        logger.error(f"Failed to get default input device: {e}")
        return None


def get_default_output_device() -> Optional[int]:
    """Get the default output device index."""
    try:
        return sd.default.device[1]
    except Exception as e:
        logger.error(f"Failed to get default output device: {e}")
        return None


def select_input_device(device_name: Optional[str] = None) -> Optional[int]:
    """
    Select input device by name (or use default).
    For built-in laptop mic, pass None to use system default.
    
    Args:
        device_name: Name of device to select, or None for default
    
    Returns:
        Device index or None if not found
    """
    if device_name is None:
        # Use default input device (built-in laptop mic)
        device_idx = get_default_input_device()
        logger.info(f"Using default input device: {device_idx}")
        return device_idx
    
    # Search for device by name
    devices = list_audio_devices()
    for device in devices:
        if device['is_input'] and device_name.lower() in device['name'].lower():
            logger.info(f"Selected input device: {device['name']} (index {device['index']})")
            return device['index']
    
    logger.warning(f"Device '{device_name}' not found, using default")
    return get_default_input_device()


def record_audio(
    duration: float,
    device: Optional[int] = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS
) -> np.ndarray:
    """
    Record audio from microphone for specified duration.
    
    Args:
        duration: Recording duration in seconds
        device: Input device index (None = default)
        sample_rate: Sample rate in Hz
        channels: Number of channels (1=mono, 2=stereo)
    
    Returns:
        numpy array of audio samples (int16)
    """
    logger.info(f"Recording {duration}s of audio from device {device}...")
    
    try:
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=DTYPE,
            device=device
        )
        
        # Wait for recording to complete
        sd.wait()
        
        logger.info(f"Recording complete: {len(audio_data)} samples")
        return audio_data
        
    except Exception as e:
        logger.error(f"Failed to record audio: {e}")
        raise


def play_audio(
    audio_data: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    device: Optional[int] = None
):
    """
    Play audio through speakers.
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: Sample rate in Hz
        device: Output device index (None = default)
    """
    logger.info(f"Playing audio: {len(audio_data)} samples at {sample_rate}Hz")
    
    try:
        sd.play(audio_data, samplerate=sample_rate, device=device)
        sd.wait()  # Wait for playback to complete
        logger.info("Playback complete")
        
    except Exception as e:
        logger.error(f"Failed to play audio: {e}")
        raise


def save_audio(
    audio_data: np.ndarray,
    filepath: str,
    sample_rate: int = SAMPLE_RATE
):
    """
    Save audio data to file.
    
    Args:
        audio_data: numpy array of audio samples
        filepath: Output file path (.wav recommended)
        sample_rate: Sample rate in Hz
    """
    logger.info(f"Saving audio to {filepath}")
    
    try:
        sf.write(filepath, audio_data, sample_rate)
        logger.info(f"Audio saved: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    logger.info(f"Loading audio from {filepath}")
    
    try:
        audio_data, sample_rate = sf.read(filepath, dtype=DTYPE)
        logger.info(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
        return audio_data, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise


class AudioStream:
    """
    Real-time audio streaming from microphone with VAD support.
    For continuous listening with voice activity detection.
    """
    
    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        device: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        chunk_size: int = CHUNK_SIZE,
        enable_vad: bool = True,
        vad_aggressiveness: int = 1,
        vad_callback: Optional[Callable[[bool, str], None]] = None,
        enable_hotword: bool = True,
        hotword_sensitivity: float = 0.7,
        hotword_callback: Optional[Callable[[], None]] = None
    ):
        """
        Initialize audio stream with optional VAD and hotword detection support.
        
        Args:
            callback: Function called with each audio chunk
            device: Input device index
            sample_rate: Sample rate in Hz
            channels: Number of channels
            chunk_size: Samples per chunk (must be 480 for VAD compatibility)
            enable_vad: Enable voice activity detection
            vad_aggressiveness: VAD sensitivity (0-3)
            vad_callback: Optional callback for VAD state changes (is_speech, state)
            enable_hotword: Enable hotword detection ("ADRIAN")
            hotword_sensitivity: Hotword detection sensitivity (0.1-1.0)
            hotword_callback: Optional callback when hotword detected
        """
        self.callback = callback
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.stream = None
        self.is_running = False
        
        # VAD support
        self.enable_vad = enable_vad and VADStateManager is not None
        self.vad_callback = vad_callback
        self.vad_manager: Optional[VADStateManager] = None
        
        if self.enable_vad:
            try:
                self.vad_manager = VADStateManager(aggressiveness=vad_aggressiveness)
                logger.info(f"VAD enabled with aggressiveness={vad_aggressiveness}")
            except Exception as e:
                logger.error(f"Failed to initialize VAD: {e}")
                self.enable_vad = False
                self.vad_manager = None
        
        if enable_vad and not self.enable_vad:
            logger.warning("VAD requested but not available - continuing without VAD")
        
        # Hotword detection support
        self.enable_hotword = enable_hotword and HotwordDetector is not None
        self.hotword_callback = hotword_callback
        self.hotword_detector: Optional[HotwordDetector] = None
        
        if self.enable_hotword:
            try:
                self.hotword_detector = HotwordDetector(
                    callback=hotword_callback,
                    sensitivity=hotword_sensitivity,
                    enable_debug=True
                )
                logger.info(f"Hotword detection enabled with sensitivity={hotword_sensitivity}")
            except Exception as e:
                logger.error(f"Failed to initialize hotword detector: {e}")
                self.enable_hotword = False
                self.hotword_detector = None
        
        if enable_hotword and not self.enable_hotword:
            logger.warning("Hotword detection requested but not available - continuing without hotword detection")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback for sounddevice stream with VAD processing."""
        # COMPREHENSIVE DEBUGGING: Track the source of frame length issues
        if status:
            logger.warning(f"[audio_utils.py:308] Audio stream status: {status}")
        
        # DEBUG: Log what we receive from sounddevice
        if not hasattr(self, '_callback_debug_counter'):
            self._callback_debug_counter = 0
        self._callback_debug_counter += 1
        
        if self._callback_debug_counter <= 3:  # Log first 3 callbacks
            logger.info(f"[audio_utils.py:305] _audio_callback #{self._callback_debug_counter} - indata.shape: {indata.shape}, frames: {frames}, dtype: {indata.dtype}")
        
        # Get audio data as mono numpy array
        audio_data = indata.copy()
        original_shape = audio_data.shape
        original_size = len(audio_data)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Take first channel if stereo
        
        # Convert to int16 for VAD processing
        if audio_data.dtype != np.int16:
            # Convert from float to int16
            max_val = np.iinfo(np.int16).max
            audio_data = (audio_data * max_val).astype(np.int16)
        
        # DEBUG: Log audio data after processing
        if self._callback_debug_counter <= 3:
            logger.info(f"[audio_utils.py:320] After processing - audio_data.shape: {audio_data.shape}, size: {len(audio_data)}, expected_chunk_size: {self.chunk_size}")
        
        # Process VAD if enabled
        if self.enable_vad and self.vad_manager:
            try:
                is_speech, vad_state = self.vad_manager.process_chunk(audio_data)
                
                # Call VAD callback if provided
                if self.vad_callback:
                    try:
                        self.vad_callback(is_speech, vad_state.value)
                    except Exception as e:
                        logger.error(f"Error in VAD callback: {e}")
                
                # Log state transitions
                if self.vad_manager.has_speech_started():
                    logger.info("Speech detected - started")
                elif self.vad_manager.has_speech_ended():
                    duration = self.vad_manager.get_speech_duration()
                    logger.info(f"Speech ended - duration: {duration:.2f}s")
                    
            except Exception as e:
                logger.error(f"Error in VAD processing: {e}")
        
        # Process hotword detection if enabled
        if self.enable_hotword and self.hotword_detector:
            try:
                # DEBUG: Log what we're sending to hotword detector
                if hasattr(self, '_callback_debug_counter') and self._callback_debug_counter <= 3:
                    logger.info(f"[audio_utils.py:364] Sending to hotword_detector.process_audio_chunk() - size: {len(audio_data)}")
                
                # Pass the audio data to hotword detector (it handles buffering internally)
                hotword_detected = self.hotword_detector.process_audio_chunk(audio_data)
                
                if hotword_detected:
                    logger.info("Hotword detected!")
                        
            except Exception as e:
                error_msg = str(e)
                # COMPREHENSIVE ERROR DEBUGGING
                logger.error(f"[audio_utils.py:377] ERROR in hotword processing - File: audio_utils.py, Line: 377, Error: {error_msg}")
                logger.error(f"[audio_utils.py:378] Context - audio_data size: {len(audio_data)}, chunk_size: {self.chunk_size}")
                logger.error(f"[audio_utils.py:379] Full exception details: {type(e).__name__}: {e}")
                
                if "Invalid frame length" in error_msg:
                    logger.error(f"[audio_utils.py:381] FRAME LENGTH ERROR - This should be handled by conversion function!")
                else:
                    logger.error(f"[audio_utils.py:384] Other error in hotword processing: {e}")
        
        # Call user callback with audio data (copy to prevent memory retention)
        try:
            # Pass a copy to prevent callback from accidentally retaining the data
            callback_data = audio_data.copy()
            self.callback(callback_data)
            # Explicitly delete to free memory immediately
            del callback_data
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        
        # Clear local audio data to free memory immediately
        del audio_data
    
    def start(self):
        """Start the audio stream."""
        if self.is_running:
            logger.warning("Audio stream already running")
            return
        
        logger.info("Starting audio stream...")
        logger.info(f"[audio_utils.py:406] Stream config - device: {self.device}, channels: {self.channels}, sample_rate: {self.sample_rate}, blocksize: {self.chunk_size}")
        
        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=DTYPE,
                callback=self._audio_callback
            )
            logger.info(f"[audio_utils.py:415] Stream created successfully with blocksize: {self.chunk_size}")
            
            self.stream.start()
            self.is_running = True
            
            # Start hotword detection if enabled
            if self.enable_hotword and self.hotword_detector:
                if self.hotword_detector.start():
                    logger.info("Audio stream and hotword detection started")
                else:
                    logger.warning("Audio stream started but hotword detection failed")
            else:
                logger.info("Audio stream started")
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def stop(self):
        """Stop the audio stream."""
        if not self.is_running:
            return
        
        logger.info("Stopping audio stream...")
        
        # Stop hotword detection first
        if self.enable_hotword and self.hotword_detector:
            self.hotword_detector.stop()
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.is_running = False
        logger.info("Audio stream stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def is_speaking(self) -> bool:
        """Check if currently detecting speech (VAD must be enabled)."""
        if not self.enable_vad or not self.vad_manager:
            return False
        return self.vad_manager.is_speaking()
    
    def get_vad_state(self) -> str:
        """Get current VAD state (VAD must be enabled)."""
        if not self.enable_vad or not self.vad_manager:
            return "disabled"
        return self.vad_manager.current_state.value
    
    def reset_vad(self):
        """Reset VAD state (VAD must be enabled)."""
        if self.enable_vad and self.vad_manager:
            self.vad_manager.reset()
            logger.debug("VAD state reset")
    
    def get_hotword_status(self) -> dict:
        """Get current hotword detection status (hotword detection must be enabled)."""
        if not self.enable_hotword or not self.hotword_detector:
            return {"enabled": False, "status": "disabled"}
        
        status = self.hotword_detector.get_status()
        status["enabled"] = True
        return status
    
    def is_hotword_enabled(self) -> bool:
        """Check if hotword detection is enabled and working."""
        return self.enable_hotword and self.hotword_detector is not None
    
    def get_memory_usage(self) -> dict:
        """Get memory usage information for monitoring."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'chunk_size_bytes': self.chunk_size * 2,  # int16 = 2 bytes per sample
            'estimated_audio_memory_mb': (32 * 2) / 1024  # ~32KB/sec for processing
        }


def get_audio_info():
    """
    Get information about the audio system.
    Useful for debugging.
    """
    info = {
        'default_input_device': get_default_input_device(),
        'default_output_device': get_default_output_device(),
        'devices': list_audio_devices(),
        'config': {
            'sample_rate': SAMPLE_RATE,
            'channels': CHANNELS,
            'dtype': DTYPE,
            'chunk_size': CHUNK_SIZE
        }
    }
    
    return info


# Convenience functions for quick testing

def quick_record(seconds: float = 5, filename: str = "recording.wav") -> str:
    """
    Quick record function for testing.
    Records from default microphone and saves to file.
    
    Args:
        seconds: Duration in seconds
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    logger.info(f"Quick record: {seconds}s to {filename}")
    
    # Record
    audio_data = record_audio(duration=seconds)
    
    # Save
    save_audio(audio_data, filename)
    
    return filename


def quick_play(filename: str):
    """
    Quick play function for testing.
    Loads and plays audio file.
    
    Args:
        filename: Audio file to play
    """
    logger.info(f"Quick play: {filename}")
    
    # Load
    audio_data, sample_rate = load_audio(filename)
    
    # Play
    play_audio(audio_data, sample_rate=sample_rate)

