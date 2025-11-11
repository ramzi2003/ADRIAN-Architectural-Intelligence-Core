"""
Shared Text-to-Speech (TTS) processor using Coqui TTS.
Can be used by both IO Service and Output Service.
Converts text responses to natural speech with JARVIS-like voice.
"""
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional, List, Callable, Dict, Any
from pathlib import Path
import threading
import queue
import time
import platform
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("shared.tts")


@dataclass
class TTSConfig:
    """TTS configuration parameters."""
    model_name: str = "tts_models/en/vctk/vits"  # Multi-speaker VITS
    speaker: str = "p234"  # Deep British male voice
    speaker_wav: Optional[str] = None  # For voice cloning (XTTS)
    language: Optional[str] = None  # Only used for XTTS voice cloning
    use_voice_cloning: bool = False  # Enable voice cloning
    speed: float = 1.0  # Speech speed multiplier
    pitch_shift: float = -2.5  # Semitones to shift pitch (negative = deeper)
    sample_rate: int = 22050
    device: str = "cpu"  # "cpu" or "cuda"
    volume: float = 1.0  # Volume multiplier (0.0 to 1.0)
    output_device_index: Optional[int] = None  # Preferred output device index
    output_device_name: Optional[str] = None  # Preferred output device name (partial match)


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
        self.enabled = True  # TTS on/off toggle
        
        # Audio playback queue
        self.playback_queue: queue.Queue = queue.Queue()
        self.playback_thread: Optional[threading.Thread] = None
        self.is_playing = False
        self._stop_playback = threading.Event()
        
        # Current playback lock (for interrupts)
        self._playback_lock = threading.Lock()
        self._current_playback_process = None
        self._playback_event_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._output_device_index: Optional[int] = None
        self.last_playback_duration_ms: float = 0.0
        self.last_playback_started_at: float = 0.0
        
        # Statistics
        self.total_synthesized = 0
        self.total_played = 0
        self.synthesis_times: List[float] = []
        
        logger.info("TTS Processor initialized")
        self._resolve_output_device()

    def _resolve_output_device(self):
        """Resolve preferred output device index based on config settings."""
        # Explicit index takes priority
        if self.config.output_device_index is not None:
            self._output_device_index = self.config.output_device_index
            logger.info(f"Configured output device index: {self._output_device_index}")
            return

        if self.config.output_device_name:
            try:
                for idx, device in enumerate(sd.query_devices()):
                    if device.get("max_output_channels", 0) > 0 and self.config.output_device_name.lower() in device.get("name", "").lower():
                        self._output_device_index = idx
                        logger.info(f"Matched output device '{device['name']}' (index {idx})")
                        return
                logger.warning(f"Output device name '{self.config.output_device_name}' not found; using default")
            except Exception as exc:
                logger.warning(f"Unable to query audio devices: {exc}")
        else:
            logger.debug("No explicit output device provided; using system default")
    
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
                os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
            
            # Set espeak-ng path for Windows
            espeak_path = r"C:\Program Files\eSpeak NG"
            if os.path.exists(espeak_path):
                os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.join(espeak_path, 'libespeak-ng.dll')
                os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.join(espeak_path, 'espeak-ng.exe')
                if espeak_path not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = espeak_path + os.pathsep + os.environ.get('PATH', '')
                logger.info(f"Set espeak-ng path: {espeak_path}")
            
            # Initialize TTS with model
            # For VITS models, don't pass language parameter
            if "vits" in self.config.model_name.lower():
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
        Apply JARVIS-like voice effects and volume control.
        
        Args:
            audio: Original audio data
            
        Returns:
            Processed audio with JARVIS effects and volume applied
        """
        try:
            from scipy import signal as scipy_signal
            from scipy.interpolate import interp1d
            
            # 1. Pitch shift (lower pitch for deeper, more authoritative voice)
            if self.config.pitch_shift != 0:
                pitch_ratio = 2 ** (self.config.pitch_shift / 12.0)
                original_length = len(audio)
                new_length = int(original_length / pitch_ratio)
                
                x_original = np.linspace(0, 1, original_length)
                x_new = np.linspace(0, 1, new_length)
                interpolator = interp1d(x_original, audio, kind='cubic', fill_value='extrapolate')
                audio_pitched = interpolator(x_new)
                audio = scipy_signal.resample(audio_pitched, original_length)
            
            # 2. Speed adjustment
            if self.config.speed != 1.0:
                new_length = int(len(audio) / self.config.speed)
                audio = scipy_signal.resample(audio, new_length)
            
            # 3. Add subtle high-pass filter (removes low rumble, cleaner sound)
            sos = scipy_signal.butter(2, 80, 'hp', fs=self.config.sample_rate, output='sos')
            audio = scipy_signal.sosfilt(sos, audio)
            
            # 4. Apply volume control
            audio = audio * self.config.volume
            
            # 5. Normalize volume (after volume control, but prevent clipping)
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.95
            
            return audio.astype(np.float32)
            
        except ImportError:
            logger.warning("scipy not available - skipping JARVIS effects")
            # Still apply volume
            audio = audio * self.config.volume
            return audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to apply JARVIS effects: {e}")
            # Still apply volume
            audio = audio * self.config.volume
            return audio.astype(np.float32)
    
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
            
            logger.debug(f"Synthesizing: '{text[:50]}...'")
            
            # Generate audio as numpy array
            if self.config.use_voice_cloning and self.config.speaker_wav:
                # Use voice cloning with reference audio
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.config.speaker_wav,
                    language=self.config.language or "en"
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
                
                win_temp = "/mnt/c/Windows/Temp"
                os.makedirs(win_temp, exist_ok=True)
                temp_file = os.path.join(win_temp, f"adrian_tts_{int(time.time()*1000)}.wav")
                
                # Save audio to WAV file
                sf.write(temp_file, audio, self.config.sample_rate)
                
                # Convert WSL path to Windows path
                win_path = temp_file.replace("/mnt/c/", "C:\\").replace("/", "\\")
                
                # Play using PowerShell (BLOCKING - wait for completion)
                with self._playback_lock:
                    self._current_playback_process = subprocess.Popen(
                        ["powershell.exe", "-Command", 
                         f"(New-Object Media.SoundPlayer '{win_path}').PlaySync(); Remove-Item '{win_path}'"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    process = self._current_playback_process
                
                # Wait for completion
                process.wait()
                
                with self._playback_lock:
                    self._current_playback_process = None
                
                logger.info(f"🔊 Audio played on Windows")
            else:
                # Native playback using sounddevice
                with self._playback_lock:
                    sd.play(
                        audio,
                        samplerate=self.config.sample_rate,
                        device=self._output_device_index
                    )
                    sd.wait()  # Wait until audio finishes playing
            
            self.total_played += 1
            logger.debug("Audio playback completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False
    
    def speak(self, text: str, blocking: bool = False, interrupt: bool = False) -> bool:
        """
        Synthesize and play text as speech.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to finish. If False, queue it.
            interrupt: If True, interrupt current speech and play immediately.
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.debug(f"TTS disabled - skipping: '{text[:50]}...'")
            return False
        
        if not text or not text.strip():
            return False
        
        if interrupt:
            # Interrupt current playback
            self.stop_playback()
            # Synthesize and play immediately
            audio = self.synthesize_speech(text)
            if audio is not None:
                return self.play_audio(audio)
            return False
        
        if blocking:
            # Synthesize and play immediately (but wait for current if playing)
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
                
                # Check if TTS is still enabled
                if not self.enabled:
                    logger.debug(f"TTS disabled - skipping queued text: '{text[:50]}...'")
                    self.playback_queue.task_done()
                    continue
                
                # Synthesize and play
                self.is_playing = True
                self.last_playback_started_at = time.perf_counter()
                self._emit_playback_event("start", {"text": text})
                audio = self.synthesize_speech(text)
                if audio is not None:
                    self.play_audio(audio)
                    self.last_playback_duration_ms = (time.perf_counter() - self.last_playback_started_at) * 1000
                    self._emit_playback_event(
                        "end",
                        {
                            "text": text,
                            "duration_ms": round(self.last_playback_duration_ms, 2),
                        }
                    )
                self.is_playing = False
                
                self.playback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback worker error: {e}")
                self.is_playing = False
                self._emit_playback_event("error", {"error": str(e)})
        
        logger.info("TTS playback worker stopped")
    
    def stop_playback(self):
        """Stop all audio playback and clear queue."""
        # Stop current playback
        sd.stop()
        
        # Stop any Windows playback process
        with self._playback_lock:
            if self._current_playback_process:
                try:
                    self._current_playback_process.terminate()
                    self._current_playback_process = None
                except:
                    pass
        
        # Clear queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except queue.Empty:
                break
        
        self.is_playing = False
        logger.info("Audio playback stopped")
        self._emit_playback_event("interrupted", {})
    
    def set_volume(self, volume: float):
        """
        Set volume level (0.0 to 1.0).
        
        Args:
            volume: Volume level (0.0 = silent, 1.0 = max)
        """
        self.config.volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to {self.config.volume:.1%}")
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable TTS.
        
        Args:
            enabled: True to enable TTS, False to disable
        """
        self.enabled = enabled
        if not enabled:
            self.stop_playback()
        logger.info(f"TTS {'enabled' if enabled else 'disabled'}")

    def set_output_device(self, *, index: Optional[int] = None, name: Optional[str] = None):
        """Update the preferred output device."""
        if index is not None:
            self.config.output_device_index = index
            self.config.output_device_name = None
        elif name:
            self.config.output_device_name = name
            self.config.output_device_index = None
        self._resolve_output_device()

    def set_voice(self, speaker_id: str):
        """Change active speaker voice."""
        self.config.speaker = speaker_id
        logger.info(f"TTS speaker set to '{speaker_id}'")

    def list_available_voices(self) -> List[str]:
        """Return available speaker IDs if supported by model."""
        try:
            if self.tts and hasattr(self.tts, "speakers"):
                return list(self.tts.speakers)
        except Exception as exc:
            logger.warning(f"Failed to list voices: {exc}")
        return []

    def set_playback_event_handler(self, handler: Optional[Callable[[str, Dict[str, Any]], None]]):
        """Register callback for playback events."""
        self._playback_event_handler = handler

    def _emit_playback_event(self, event: str, payload: Dict[str, Any]):
        if not self._playback_event_handler:
            return
        try:
            self._playback_event_handler(event, payload)
        except Exception as exc:
            logger.debug(f"Playback event handler error: {exc}")
    
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
            "enabled": self.enabled,
            "volume": self.config.volume,
            "total_synthesized": self.total_synthesized,
            "total_played": self.total_played,
            "avg_synthesis_time": round(avg_synthesis_time, 2),
            "queue_size": self.playback_queue.qsize(),
            "is_playing": self.is_playing,
            "speed": self.config.speed,
            "sample_rate": self.config.sample_rate,
            "output_device_index": self._output_device_index,
            "last_playback_duration_ms": round(self.last_playback_duration_ms, 2),
        }

