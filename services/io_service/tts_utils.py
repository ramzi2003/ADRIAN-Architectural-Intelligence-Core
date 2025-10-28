"""
Text-to-Speech (TTS) utilities using Coqui TTS.
Converts text responses to natural speech with JARVIS-like voice.
"""
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional, List
from pathlib import Path
import threading
import queue
import time
import platform
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("io-service")


@dataclass
class TTSConfig:
    """TTS configuration parameters."""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"  # XTTS for voice cloning
    vocoder_name: Optional[str] = None  # Auto-selected
    speaker: Optional[str] = None  # Not used for cloned voice
    speaker_wav: Optional[str] = "data/jarvis_cloned_test.wav"  # Use the cloned JARVIS voice
    language: Optional[str] = None  # Only used for XTTS voice cloning
    use_voice_cloning: bool = True  # Enable JARVIS voice cloning
    speed: float = 1.15  # Faster for efficient AI delivery (JARVIS-like)
    pitch_shift: float = -2.5  # Semitones to shift pitch (negative = deeper)
    sample_rate: int = 22050
    device: str = "cpu"  # "cpu" or "cuda"


class TTSProcessor:
    """
    Handles text-to-speech conversion using Coqui TTS.
    Supports audio playback queue for multiple messages.
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize TTS processor.
        
        Args:
            config: TTS configuration (uses defaults if None)
        """
        self.config = config or TTSConfig()
        self.tts = None
        self.is_model_loaded = False
        
        # Audio playback queue
        self.playback_queue: queue.Queue = queue.Queue()
        self.playback_thread: Optional[threading.Thread] = None
        self.is_playing = False
        self._stop_playback = threading.Event()
        
        # Statistics
        self.total_synthesized = 0
        self.total_played = 0
        self.synthesis_times: List[float] = []
        
        logger.info("TTS Processor initialized")
    
    def load_model(self) -> bool:
        """
        Load the TTS model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading TTS model: {self.config.model_name}")
            
            from TTS.api import TTS
            import torch
            import os
            
            # Fix PyTorch weights loading for XTTS
            if "xtts" in self.config.model_name.lower():
                # Disable weights_only for XTTS compatibility
                os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
            
            # Set espeak-ng path for Windows
            espeak_path = r"C:\Program Files\eSpeak NG"
            if os.path.exists(espeak_path):
                os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.join(espeak_path, 'libespeak-ng.dll')
                os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.join(espeak_path, 'espeak-ng.exe')
                # Add to PATH
                if espeak_path not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = espeak_path + os.pathsep + os.environ.get('PATH', '')
                logger.info(f"Set espeak-ng path: {espeak_path}")
            
            # Initialize TTS with model
            # For VITS models, don't pass language parameter
            if "vits" in self.config.model_name.lower():
                # Temporarily remove language for VITS models
                original_language = self.config.language
                self.config.language = None
                
            self.tts = TTS(
                model_name=self.config.model_name,
                progress_bar=True,
                gpu=(self.config.device == "cuda")
            )
            
            # Restore language for XTTS models
            if "vits" in self.config.model_name.lower():
                self.config.language = original_language
            
            self.is_model_loaded = True
            logger.info(f"✅ TTS model loaded: {self.config.model_name}")
            
            # Start playback thread
            self._start_playback_thread()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            
            # Try simpler model as fallback
            try:
                from TTS.api import TTS
                logger.info("Trying simpler model: tts_models/en/ljspeech/glow-tts")
                self.tts = TTS(
                    model_name="tts_models/en/ljspeech/glow-tts",
                    progress_bar=True,
                    gpu=(self.config.device == "cuda")
                )
                # Disable speaker for single-speaker models
                self.config.speaker = None
                self.is_model_loaded = True
                logger.info(f"✅ TTS model loaded (glow-tts)")
                self._start_playback_thread()
                return True
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                self.is_model_loaded = False
                return False
    
    def _apply_jarvis_effects(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply JARVIS-like voice effects to make it sound more like an AI butler.
        
        Args:
            audio: Original audio data
            
        Returns:
            Processed audio with JARVIS effects
        """
        try:
            from scipy import signal as scipy_signal
            from scipy.interpolate import interp1d
            
            # 1. Pitch shift (lower pitch for deeper, more authoritative voice)
            if self.config.pitch_shift != 0:
                # Calculate pitch shift ratio
                pitch_ratio = 2 ** (self.config.pitch_shift / 12.0)
                
                # Resample to shift pitch
                original_length = len(audio)
                new_length = int(original_length / pitch_ratio)
                
                # Create interpolation function
                x_original = np.linspace(0, 1, original_length)
                x_new = np.linspace(0, 1, new_length)
                interpolator = interp1d(x_original, audio, kind='cubic', fill_value='extrapolate')
                audio_pitched = interpolator(x_new)
                
                # Resample back to original length to maintain speed
                audio = scipy_signal.resample(audio_pitched, original_length)
            
            # 2. Speed adjustment (slightly slower for butler-like delivery)
            if self.config.speed != 1.0:
                new_length = int(len(audio) / self.config.speed)
                audio = scipy_signal.resample(audio, new_length)
            
            # 3. Add subtle high-pass filter (removes low rumble, cleaner sound)
            sos = scipy_signal.butter(2, 80, 'hp', fs=self.config.sample_rate, output='sos')
            audio = scipy_signal.sosfilt(sos, audio)
            
            # 4. Normalize volume
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.95
            
            return audio.astype(np.float32)
            
        except ImportError:
            logger.warning("scipy not available - skipping JARVIS effects")
            return audio
        except Exception as e:
            logger.warning(f"Failed to apply JARVIS effects: {e}")
            return audio
    
    def synthesize_speech(self, text: str) -> Optional[np.ndarray]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array, or None if failed
        """
        if not self.is_model_loaded:
            logger.error("TTS model not loaded")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        try:
            start_time = time.time()
            
            # Synthesize speech
            logger.debug(f"Synthesizing: '{text[:50]}...'")
            
            # Generate audio as numpy array
            if self.config.use_voice_cloning and self.config.speaker_wav:
                # Use voice cloning with JARVIS reference audio
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.config.speaker_wav,
                    language=self.config.language or "en"  # Default to English for XTTS
                )
            elif self.config.speaker:
                # Use pre-trained speaker (VITS model) - no language parameter needed
                audio = self.tts.tts(text=text, speaker=self.config.speaker)
            else:
                # Default synthesis
                audio = self.tts.tts(text=text)
            
            # Convert to numpy array if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Apply JARVIS-like post-processing effects
            audio = self._apply_jarvis_effects(audio)
            
            synthesis_time = time.time() - start_time
            self.synthesis_times.append(synthesis_time)
            self.total_synthesized += 1
            
            logger.info(f"🎙️ Synthesized speech: '{text[:50]}...' ({synthesis_time:.2f}s)")
            
            return audio
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None
    
    def play_audio(self, audio: np.ndarray) -> bool:
        """
        Play audio through speakers (or save to file if in WSL).
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure audio is in correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio to [-1, 1] range
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Check if running in WSL
            is_wsl = 'microsoft' in platform.uname().release.lower()
            
            if is_wsl:
                # In WSL: save to Windows temp and play with PowerShell
                import tempfile
                import os
                
                # Create temp file on Windows side
                win_temp = "/mnt/c/Windows/Temp"
                os.makedirs(win_temp, exist_ok=True)
                temp_file = os.path.join(win_temp, f"adrian_tts_{int(time.time()*1000)}.wav")
                
                # Save audio to WAV file
                sf.write(temp_file, audio, self.config.sample_rate)
                
                # Convert WSL path to Windows path
                win_path = temp_file.replace("/mnt/c/", "C:\\").replace("/", "\\")
                
                # Play using PowerShell (BLOCKING - wait for completion)
                result = subprocess.run(
                    ["powershell.exe", "-Command", 
                     f"(New-Object Media.SoundPlayer '{win_path}').PlaySync(); Remove-Item '{win_path}'"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
                
                logger.info(f"🔊 Audio played on Windows: {win_path}")
            else:
                # Native playback using sounddevice
                sd.play(audio, samplerate=self.config.sample_rate)
                sd.wait()  # Wait until audio finishes playing
            
            self.total_played += 1
            logger.debug("Audio playback completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False
    
    def speak(self, text: str, blocking: bool = False) -> bool:
        """
        Synthesize and play text as speech.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to finish. If False, queue it.
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False
        
        if blocking:
            # Synthesize and play immediately
            audio = self.synthesize_speech(text)
            if audio is not None:
                return self.play_audio(audio)
            return False
        else:
            # Add to queue for async playback
            self.playback_queue.put(text)
            return True
    
    def _start_playback_thread(self):
        """Start the background playback thread."""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self._stop_playback.clear()
            self.playback_thread = threading.Thread(
                target=self._playback_worker,
                daemon=True,
                name="TTS-Playback"
            )
            self.playback_thread.start()
            logger.info("TTS playback thread started")
    
    def _playback_worker(self):
        """Background worker that processes the playback queue."""
        logger.info("TTS playback worker running")
        
        while not self._stop_playback.is_set():
            try:
                # Get text from queue (with timeout to check stop flag)
                text = self.playback_queue.get(timeout=0.5)
                
                # Synthesize and play
                self.is_playing = True
                audio = self.synthesize_speech(text)
                if audio is not None:
                    self.play_audio(audio)
                self.is_playing = False
                
                self.playback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback worker error: {e}")
                self.is_playing = False
        
        logger.info("TTS playback worker stopped")
    
    def stop_playback(self):
        """Stop all audio playback and clear queue."""
        # Stop current playback
        sd.stop()
        
        # Clear queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except queue.Empty:
                break
        
        self.is_playing = False
        logger.info("Audio playback stopped")
    
    def shutdown(self):
        """Shutdown TTS processor and cleanup resources."""
        logger.info("Shutting down TTS processor...")
        
        # Stop playback thread
        self._stop_playback.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        # Stop any playing audio
        self.stop_playback()
        
        # Clear model
        self.tts = None
        self.is_model_loaded = False
        
        logger.info("TTS processor shutdown complete")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available TTS models.
        
        Returns:
            List of model names
        """
        try:
            from TTS.api import TTS
            return TTS.list_models()
        except Exception as e:
            logger.error(f"Failed to list TTS models: {e}")
            return []
    
    def get_performance_stats(self) -> dict:
        """
        Get TTS performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_synthesis_time = (
            sum(self.synthesis_times) / len(self.synthesis_times)
            if self.synthesis_times else 0.0
        )
        
        return {
            "model_loaded": self.is_model_loaded,
            "model_name": self.config.model_name,
            "total_synthesized": self.total_synthesized,
            "total_played": self.total_played,
            "avg_synthesis_time": round(avg_synthesis_time, 2),
            "queue_size": self.playback_queue.qsize(),
            "is_playing": self.is_playing,
            "speed": self.config.speed,
            "sample_rate": self.config.sample_rate
        }


# Convenience function for quick TTS
_global_tts: Optional[TTSProcessor] = None

def text_to_speech(text: str, blocking: bool = False) -> bool:
    """
    Convert text to speech (convenience function).
    Uses a global TTS processor instance.
    
    Args:
        text: Text to speak
        blocking: If True, wait for speech to finish
        
    Returns:
        True if successful, False otherwise
    """
    global _global_tts
    
    if _global_tts is None:
        _global_tts = TTSProcessor()
        if not _global_tts.load_model():
            return False
    
    return _global_tts.speak(text, blocking=blocking)

