"""
Hotword Detection utilities for ADRIAN IO Service.
Uses Porcupine for "ADRIAN" wake word detection.
"""
import time
import asyncio
import os
from typing import Optional, Callable, Dict, Any
import numpy as np
from pathlib import Path
import json

from shared.logging_config import get_logger

# Try to import Porcupine
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    pvporcupine = None
    PORCUPINE_AVAILABLE = False

logger = get_logger("io-service.hotword")

# Hotword configuration
SAMPLE_RATE = 16000  # Hz - Must match Porcupine requirements
CHUNK_SIZE = 512  # Samples per chunk for hotword detection (32ms at 16kHz)


class HotwordDetector:
    """
    Hotword Detection using Porcupine.
    Listens for "ADRIAN" wake word with smart error handling.
    """
    
    def __init__(
        self,
        callback: Optional[Callable[[], None]] = None,
        sensitivity: float = 0.7,
        cooldown_period: float = 2.5,
        enable_debug: bool = True
    ):
        """
        Initialize hotword detector.
        
        Args:
            callback: Function to call when "ADRIAN" is detected
            sensitivity: Detection sensitivity (0.1-1.0)
            cooldown_period: Minimum seconds between detections
            enable_debug: Enable debug logging
        """
        self.callback = callback
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.cooldown_period = cooldown_period
        self.enable_debug = enable_debug
        
        # State tracking
        self.is_running = False
        self.last_detection_time = 0.0
        self.detection_count = 0
        self.failure_count = 0
        self.max_failures = 3
        
        # Audio processing configuration
        self.chunk_size = CHUNK_SIZE  # Will be updated based on the model
        
        # Porcupine instances
        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self.porcupine_keywords = ["adrian"]  # We'll start with "hey google" for testing
        self.offline_mode = False  # Track if running in offline mode
        
        # Error handling
        self.error_context: Dict[str, Any] = {}
        
        logger.info(f"HotwordDetector initialized: sensitivity={sensitivity}, cooldown={cooldown_period}s")
    
    def _get_adrian_model_path(self) -> Optional[str]:
        """
        Get the path to the custom ADRIAN model file.
        
        Returns:
            Path to the .ppn model file if it exists, None otherwise
        """
        # Look for the model file in common locations
        possible_paths = [
            # In the project data directory
            Path(__file__).parent.parent.parent / "data" / "adrian.ppn",
            Path(__file__).parent.parent.parent / "models" / "adrian.ppn",
            # In the current directory
            Path(__file__).parent / "adrian.ppn",
            # In user's home directory
            Path.home() / ".adrian" / "models" / "adrian.ppn",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found ADRIAN model at: {path}")
                return str(path)
        
        logger.debug("No custom ADRIAN model file found")
        return None
    
    def _initialize_porcupine(self) -> bool:
        """
        Initialize Porcupine with available keywords.
        Will try custom "ADRIAN" model first, fallback to built-in models.
        """
        if not PORCUPINE_AVAILABLE:
            logger.error("Porcupine not available. Install with: pip install pvporcupine")
            return False
        
        try:
            # Get access key from settings
            from shared.config import get_settings
            import os
            
            # DEBUG: Check environment variable directly first
            direct_access_key = os.getenv('PICOVOICE_ACCESS_KEY')
            logger.info(f"[hotword_utils.py:119] Direct env var PICOVOICE_ACCESS_KEY: {bool(direct_access_key)}")
            
            settings = get_settings()
            access_key = settings.picovoice_access_key
            
            # DEBUG: Check if access key is loaded
            logger.info(f"[hotword_utils.py:122] Loading settings - Access key loaded: {bool(access_key)}")
            if access_key:
                logger.info(f"[hotword_utils.py:123] Access key length: {len(access_key)} characters")
            else:
                logger.warning(f"[hotword_utils.py:124] No access key found in settings!")
                
            # Try to use direct env var if settings failed
            if not access_key and direct_access_key:
                access_key = direct_access_key
                logger.info(f"[hotword_utils.py:127] Using direct environment variable for access key")
            
            # For testing without API key, we'll use a simpler approach
            if not access_key:
                logger.warning("No PICOVOICE_ACCESS_KEY found. Using demo mode for testing.")
                logger.info("Porcupine requires an access key. We'll use built-in keywords.")
                
                # Try to use built-in keywords that might work without key
                try:
                    # Some versions of Porcupine allow built-in keywords without key
                    self.porcupine = pvporcupine.create(
                        keywords=["computer"],  # Built-in keyword
                        sensitivities=[self.sensitivity]
                    )
                    self.porcupine_keywords = ["computer"]
                    
                    # Get the actual frame size for demo mode too
                    try:
                        frame_length = self.porcupine.frame_length
                        if frame_length != self.chunk_size:
                            logger.info(f"Demo mode: updating chunk size from {self.chunk_size} to {frame_length}")
                            self.chunk_size = frame_length
                    except AttributeError:
                        logger.debug("Could not get frame_length from Porcupine instance in demo mode")
                    
                    logger.info("Porcupine initialized in demo mode with 'computer' keyword")
                    return True
                except Exception as demo_e:
                    logger.info(f"Demo mode failed: {demo_e}")
                    logger.info("Hotword detection disabled. You can still test other features.")
                    return False  # Gracefully disable hotword
            
            # First, try to load custom ADRIAN model if it exists
            custom_model_path = self._get_adrian_model_path()
            if custom_model_path and os.path.exists(custom_model_path):
                try:
                    # Custom models (.ppn files) require access key for initialization
                    # but once loaded, they work completely offline
                    if access_key:
                        self.porcupine = pvporcupine.create(
                            keyword_paths=[custom_model_path],
                            sensitivities=[self.sensitivity],
                            access_key=access_key
                        )
                        self.offline_mode = True  # Model works offline after initialization
                        logger.info("🎉 Custom ADRIAN model loaded! Works offline after initialization.")
                    else:
                        raise Exception("Access key required for custom model initialization")
                    self.porcupine_keywords = ["adrian"]  # Custom model
                    
                    # Get the actual frame size from the Porcupine instance
                    # Custom models might have different frame sizes
                    try:
                        frame_length = self.porcupine.frame_length
                        logger.info(f"[hotword_utils.py:169] Porcupine reports frame_length: {frame_length} - File: hotword_utils.py, Line: 169")
                        logger.info(f"[hotword_utils.py:170] Current self.chunk_size: {self.chunk_size}")
                        if frame_length != self.chunk_size:
                            logger.warning(f"[hotword_utils.py:172] CUSTOM MODEL SIZE MISMATCH - File: hotword_utils.py, Line: 172")
                            logger.warning(f"[hotword_utils.py:173] Custom model expects {frame_length} samples, but we were using {self.chunk_size}")
                            # Update our chunk size to match the model
                            old_chunk_size = self.chunk_size
                            self.chunk_size = frame_length
                            logger.info(f"[hotword_utils.py:175] Updated chunk_size from {old_chunk_size} to {self.chunk_size} for custom model")
                        else:
                            logger.info(f"[hotword_utils.py:177] Chunk size already correct: {self.chunk_size}")
                    except AttributeError:
                        logger.warning(f"[hotword_utils.py:179] Could not get frame_length from Porcupine instance - File: hotword_utils.py, Line: 179")
                        logger.info(f"[hotword_utils.py:180] Using default chunk_size: {self.chunk_size}")
                    
                    logger.info("🎉 Loaded custom ADRIAN wake word model!")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load custom ADRIAN model: {e}")
            
            # Fallback to built-in keywords
            keyword_attempts = [
                ["jarvis"],           # Similar name to ADRIAN (temporary fallback)
                ["computer"],         # Built-in keyword
                ["hey google"],       # Built-in model  
            ]
            
            for keywords in keyword_attempts:
                try:
                    self.porcupine = pvporcupine.create(
                        keywords=keywords,
                        sensitivities=[self.sensitivity] * len(keywords),
                        access_key=access_key
                    )
                    self.porcupine_keywords = keywords
                    
                    # Get the actual frame size from the Porcupine instance
                    try:
                        frame_length = self.porcupine.frame_length
                        if frame_length != self.chunk_size:
                            logger.info(f"Porcupine model expects {frame_length} samples, updating from {self.chunk_size}")
                            self.chunk_size = frame_length
                    except AttributeError:
                        logger.debug("Could not get frame_length from Porcupine instance")
                    
                    if keywords == ["jarvis"]:
                        logger.info(f"Porcupine initialized with '{keywords[0]}' keyword")
                        logger.info("💡 Note: Say 'JARVIS' instead of 'ADRIAN' for now - sounds similar!")
                        logger.info("💡 To use 'ADRIAN', create a custom model at https://console.picovoice.ai")
                    elif keywords == ["adrian"]:
                        logger.info("🎉 Porcupine initialized with custom 'ADRIAN' keyword!")
                    else:
                        logger.info(f"Porcupine initialized with keywords: {keywords}")
                    return True
                    
                except Exception as e:
                    logger.debug(f"Failed to initialize with {keywords}: {e}")
                    continue
            
            logger.error("Failed to initialize Porcupine with any keywords")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing Porcupine: {e}")
            self.error_context = {"error": str(e), "stage": "initialization"}
            return False
    
    def start(self) -> bool:
        """Start hotword detection."""
        if self.is_running:
            logger.warning("Hotword detection already running")
            return True
        
        if not self._initialize_porcupine():
            logger.error("Cannot start hotword detection - Porcupine initialization failed")
            return False
        
        self.is_running = True
        self.detection_count = 0
        self.failure_count = 0
        logger.info("Hotword detection started")
        return True
    
    def stop(self):
        """Stop hotword detection."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.porcupine:
            try:
                self.porcupine.delete()
                logger.debug("Porcupine instance deleted")
            except Exception as e:
                logger.error(f"Error deleting Porcupine instance: {e}")
        
        self.porcupine = None
        logger.info("Hotword detection stopped")
    
    def _convert_to_correct_chunk_size(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert any input audio chunk to the exact size expected by Porcupine.
        
        Args:
            audio_data: Input audio data of any size
            
        Returns:
            Audio data resized to self.chunk_size
        """
        # COMPREHENSIVE DEBUGGING: Track conversion function
        if not hasattr(self, '_convert_debug_counter'):
            self._convert_debug_counter = 0
        self._convert_debug_counter += 1
        
        if self._convert_debug_counter <= 3:
            logger.info(f"[hotword_utils.py:283] _convert_to_correct_chunk_size #{self._convert_debug_counter} - File: hotword_utils.py, Line: 283")
            logger.info(f"[hotword_utils.py:284] Input - size: {len(audio_data)}, dtype: {audio_data.dtype}, shape: {audio_data.shape}")
            logger.info(f"[hotword_utils.py:285] Target chunk_size: {self.chunk_size}")
        
        if len(audio_data) == 0:
            logger.warning(f"[hotword_utils.py:290] Empty audio data in conversion function")
            return np.array([], dtype=np.int16)
        
        # Ensure int16 format
        original_dtype = audio_data.dtype
        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        audio_data = audio_data.flatten()
        
        target_size = self.chunk_size
        
        if self._convert_debug_counter <= 3:
            logger.info(f"[hotword_utils.py:301] After flatten - size: {len(audio_data)}, target_size: {target_size}")
        
        if len(audio_data) == target_size:
            result = audio_data.copy()
            if self._convert_debug_counter <= 3:
                logger.info(f"[hotword_utils.py:306] CASE: Exact size match - no change needed")
        elif len(audio_data) > target_size:
            # Take first target_size samples
            result = audio_data[:target_size].copy()
            if self._convert_debug_counter <= 3:
                logger.info(f"[hotword_utils.py:310] CASE: Truncating from {len(audio_data)} to {target_size}")
        else:
            # Pad with zeros to reach target size
            padding_needed = target_size - len(audio_data)
            result = np.concatenate([audio_data, np.zeros(padding_needed, dtype=np.int16)])
            if self._convert_debug_counter <= 3:
                logger.info(f"[hotword_utils.py:315] CASE: Padding from {len(audio_data)} to {target_size}")
        
        # Final safety check
        if len(result) != target_size:
            logger.error(f"[hotword_utils.py:318] CRITICAL: Conversion safety check failed - File: hotword_utils.py, Line: 318")
            logger.error(f"[hotword_utils.py:319] Got {len(result)}, expected {target_size}")
            # Force correct size as last resort
            if len(result) > target_size:
                result = result[:target_size]
                logger.error(f"[hotword_utils.py:322] Forced truncation to {len(result)}")
            else:
                padding = np.zeros(target_size - len(result), dtype=np.int16)
                result = np.concatenate([result, padding])
                logger.error(f"[hotword_utils.py:325] Forced padding to {len(result)}")
        
        if self._convert_debug_counter <= 3:
            logger.info(f"[hotword_utils.py:328] FINAL RESULT - size: {len(result)}, target: {target_size}, match: {len(result) == target_size}")
        
        return result
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """
        Process audio chunk for hotword detection.
        Converts input to correct chunk size before processing.
        
        Args:
            audio_data: Audio chunk as numpy array (int16, mono, 16kHz)
            
        Returns:
            True if hotword detected, False otherwise
        """
        # COMPREHENSIVE DEBUGGING: Track the exact issue
        if not hasattr(self, '_process_debug_counter'):
            self._process_debug_counter = 0
        self._process_debug_counter += 1
        
        if self._process_debug_counter <= 3:  # Log first 3 calls
            logger.info(f"[hotword_utils.py:373] process_audio_chunk #{self._process_debug_counter} - File: hotword_utils.py, Line: 373")
            logger.info(f"[hotword_utils.py:374] Input - audio_data.shape: {audio_data.shape}, size: {len(audio_data)}, dtype: {audio_data.dtype}")
            logger.info(f"[hotword_utils.py:375] Expected chunk_size: {self.chunk_size}, is_running: {self.is_running}, porcupine: {self.porcupine is not None}")
            logger.info(f"[hotword_utils.py:376] ROOT CAUSE CHECK: Input size {len(audio_data)} -> should convert to {self.chunk_size}")
        
        if not self.is_running or not self.porcupine:
            if self._process_debug_counter <= 3:
                logger.warning(f"[hotword_utils.py:340] Early return - not running or no porcupine")
            return False
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_detection_time < self.cooldown_period:
            if self._process_debug_counter <= 3:
                logger.debug(f"[hotword_utils.py:345] In cooldown period")
            return False
        
        if len(audio_data) == 0:
            if self._process_debug_counter <= 3:
                logger.warning(f"[hotword_utils.py:349] Empty audio data")
            return False
        
        try:
            # CRITICAL DEBUG: Log what we receive before any processing
            if not hasattr(self, '_conversion_log_counter'):
                self._conversion_log_counter = 0
            self._conversion_log_counter += 1
            
            if self._conversion_log_counter <= 5:  # Log first 5 conversions in detail
                logger.info(f"[hotword_utils.py:397] BEFORE CONVERSION - Input: {len(audio_data)} samples")
            
            # Convert to the exact size Porcupine expects
            processed_chunk = self._convert_to_correct_chunk_size(audio_data)
            
            if self._conversion_log_counter <= 5:  # Log first 5 conversions in detail
                logger.info(f"[hotword_utils.py:403] AFTER CONVERSION - Input: {len(audio_data)}, Output: {len(processed_chunk)}, Target: {self.chunk_size}")
                logger.info(f"[hotword_utils.py:404] CONVERSION VERIFICATION: {len(processed_chunk) == self.chunk_size}")
            
            if len(processed_chunk) != self.chunk_size:
                logger.error(f"[hotword_utils.py:364] CRITICAL CONVERSION FAILURE - File: hotword_utils.py, Line: 364")
                logger.error(f"[hotword_utils.py:365] Got {len(processed_chunk)}, expected {self.chunk_size}")
                return False
            
            # Convert to bytes for Porcupine - ensure correct format
            # CRITICAL: Make sure processed_chunk is exactly the right size
            processed_chunk = processed_chunk.astype(np.int16)
            
            # Final safety check - this should never fail, but let's be extra sure
            if len(processed_chunk) != self.chunk_size:
                logger.error(f"[hotword_utils.py:418] CRITICAL: Size mismatch before bytes conversion!")
                logger.error(f"[hotword_utils.py:419] processed_chunk: {len(processed_chunk)}, expected: {self.chunk_size}")
                # Force the correct size as absolute last resort
                if len(processed_chunk) > self.chunk_size:
                    processed_chunk = processed_chunk[:self.chunk_size]
                else:
                    # Pad with zeros
                    padding = np.zeros(self.chunk_size - len(processed_chunk), dtype=np.int16)
                    processed_chunk = np.concatenate([processed_chunk, padding])
                logger.warning(f"[hotword_utils.py:426] Forced size correction to {len(processed_chunk)}")
            
            # CRITICAL FIX: Porcupine expects numpy array of samples, NOT bytes!
            # The error "expected 512 but received 1024" suggests Porcupine is counting
            # the input as individual elements, so bytes (1024) vs samples (512)
            
            # Verify we have exactly the right number of samples
            if len(processed_chunk) != self.chunk_size:
                logger.error(f"[hotword_utils.py:432] CRITICAL: Sample count mismatch - File: hotword_utils.py, Line: 432")
                logger.error(f"[hotword_utils.py:433] {len(processed_chunk)} samples != {self.chunk_size} expected")
                return False
            
            # DEBUG: Log before Porcupine process call
            if self._conversion_log_counter <= 5:
                logger.info(f"[hotword_utils.py:436] About to call porcupine.process() - File: hotword_utils.py, Line: 436")
                logger.info(f"[hotword_utils.py:437] processed_chunk: {len(processed_chunk)} samples, dtype: {processed_chunk.dtype}")
                logger.info(f"[hotword_utils.py:438] Expected: {self.chunk_size} samples")
                logger.info(f"[hotword_utils.py:439] porcupine.frame_length: {getattr(self.porcupine, 'frame_length', 'unknown')}")
            
            # Process with Porcupine - FIXED: Pass numpy array directly
            try:
                # FINAL DEBUG: Log exactly what we're sending to Porcupine
                if self._conversion_log_counter <= 10:  # Log first 10 attempts
                    logger.info(f"[hotword_utils.py:444] FINAL CHECK before porcupine.process() - FIXED VERSION")
                    logger.info(f"[hotword_utils.py:445] processed_chunk samples: {len(processed_chunk)}, dtype: {processed_chunk.dtype}")
                    logger.info(f"[hotword_utils.py:446] Expected: {self.chunk_size} samples")
                    logger.info(f"[hotword_utils.py:447] porcupine.frame_length: {getattr(self.porcupine, 'frame_length', 'unknown')}")
                
                # CRITICAL FIX: Ensure proper numpy array format
                # Make sure it's exactly what Porcupine expects
                processed_chunk = processed_chunk.astype(np.int16).flatten()
                
                # Final verification
                if len(processed_chunk) != self.chunk_size:
                    logger.error(f"[hotword_utils.py:477] CRITICAL: Final size check failed!")
                    logger.error(f"[hotword_utils.py:478] Got {len(processed_chunk)}, expected {self.chunk_size}")
                    return False
                
                if self._conversion_log_counter <= 3:
                    logger.info(f"[hotword_utils.py:480] FINAL CHECK - Passing to Porcupine:")
                    logger.info(f"[hotword_utils.py:481] processed_chunk: {len(processed_chunk)} samples, dtype: {processed_chunk.dtype}")
                    logger.info(f"[hotword_utils.py:482] Shape: {processed_chunk.shape}")
                    logger.info(f"[hotword_utils.py:483] First 5 values: {processed_chunk[:5]}")
                
                # Try numpy array first, fallback to bytes if needed
                try:
                    keyword_index = self.porcupine.process(processed_chunk)
                    if self._conversion_log_counter <= 3:
                        logger.info(f"[hotword_utils.py:487] SUCCESS with numpy array format!")
                except Exception as numpy_error:
                    if "Invalid frame length" in str(numpy_error):
                        if self._conversion_log_counter <= 3:
                            logger.info(f"[hotword_utils.py:491] NumPy format failed, trying bytes format as fallback")
                        # Fallback to bytes (though this shouldn't be needed)
                        audio_bytes = processed_chunk.tobytes()
                        keyword_index = self.porcupine.process(audio_bytes)
                        if self._conversion_log_counter <= 3:
                            logger.info(f"[hotword_utils.py:495] SUCCESS with bytes format fallback!")
                    else:
                        raise numpy_error
                if self._conversion_log_counter <= 5:
                    logger.info(f"[hotword_utils.py:460] Porcupine.process() succeeded, keyword_index: {keyword_index}")
            except Exception as porcupine_error:
                logger.error(f"[hotword_utils.py:462] PORCUPINE PROCESS ERROR - File: hotword_utils.py, Line: 462")
                logger.error(f"[hotword_utils.py:463] Error type: {type(porcupine_error).__name__}, Message: {str(porcupine_error)}")
                logger.error(f"[hotword_utils.py:464] Context - audio_bytes: {len(audio_bytes)} bytes, processed_chunk: {len(processed_chunk)} samples")
                logger.error(f"[hotword_utils.py:465] Expected chunk_size: {self.chunk_size}, porcupine.frame_length: {getattr(self.porcupine, 'frame_length', 'unknown')}")
                
                # CRITICAL DEBUG: Let's see what the actual byte content looks like
                if self._conversion_log_counter <= 3:
                    logger.error(f"[hotword_utils.py:467] DEBUG: First few bytes of audio_bytes: {audio_bytes[:16]}")
                    logger.error(f"[hotword_utils.py:468] DEBUG: processed_chunk array: {processed_chunk[:10]}")
                
                raise  # Re-raise to be caught by outer try-catch
            
            if keyword_index >= 0:
                # Hotword detected!
                self.last_detection_time = current_time
                self.detection_count += 1
                
                detected_keyword = self.porcupine_keywords[keyword_index] if keyword_index < len(self.porcupine_keywords) else "unknown"
                logger.info(f"🎯 Hotword detected: '{detected_keyword}' (detection #{self.detection_count})")
                
                # Call callback
                if self.callback:
                    try:
                        self.callback()
                        self.failure_count = 0  # Reset failure count on success
                    except Exception as e:
                        logger.error(f"Error in hotword callback: {e}")
                        self._handle_detection_crash(e)
                
                return True
                
        except Exception as e:
            error_msg = str(e)
            if "Invalid frame length" in error_msg:
                # This should NOT happen with direct conversion approach
                logger.error(f"Frame length error with conversion approach - this indicates deeper issue: {e}")
            else:
                logger.error(f"Error processing hotword chunk: {e}")
        
        return False
    
    def _handle_detection_crash(self, error: Exception):
        """
        Handle crash after hotword detection (smart error handling).
        This will generate a personality-driven response instead of a script.
        """
        logger.error(f"Hotword detection crash: {error}")
        
        # Store error context for personality-driven response
        self.error_context.update({
            "error": str(error),
            "stage": "post_detection_crash",
            "user_triggered": True,  # User said "ADRIAN" so they were trying to talk
        })
        
        # Trigger personality-driven error response
        # This will be handled by the main service to generate natural responses
        asyncio.create_task(self._request_error_response())
    
    async def _request_error_response(self):
        """Request a personality-driven error response from the processing service."""
        try:
            # For now, we'll emit an event that the main service can handle
            # Later, this will connect to the processing service for natural responses
            logger.info("Requesting personality-driven error response for post-detection crash")
            
            # TODO: Connect to processing service or Redis to request natural response
            # For now, just log the context for manual handling
            
        except Exception as e:
            logger.error(f"Error requesting error response: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of hotword detector."""
        return {
            "is_running": self.is_running,
            "porcupine_available": PORCUPINE_AVAILABLE,
            "keywords": self.porcupine_keywords,
            "sensitivity": self.sensitivity,
            "detection_count": self.detection_count,
            "failure_count": self.failure_count,
            "last_detection": self.last_detection_time,
            "cooldown_remaining": max(0, self.cooldown_period - (time.time() - self.last_detection_time)),
            "offline_mode": self.offline_mode,
            "chunk_size": self.chunk_size
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_adrian_model() -> bool:
    """
    Create custom ADRIAN wake word model.
    
    For now, this provides instructions on how to create the model.
    Eventually, this will automate the process.
    
    Returns:
        True if model creation is successful or already exists
    """
    logger.info("Custom ADRIAN model creation:")
    logger.info("1. Go to: https://picovoice.ai/docs/quick-start/porcupine-python/")
    logger.info("2. Use Picovoice Console to create custom 'ADRIAN' keyword")
    logger.info("3. Download the .ppn file and place it in the project")
    logger.info("4. Update the code to use the custom model file")
    
    # TODO: Implement automatic model creation
    return False


def get_adrian_model_info() -> dict:
    """
    Get information about creating the ADRIAN model.
    
    Returns:
        Dictionary with instructions and status
    """
    # Check if custom model exists
    custom_paths = [
        Path(__file__).parent.parent.parent / "data" / "adrian.ppn",
        Path(__file__).parent.parent.parent / "models" / "adrian.ppn",
        Path(__file__).parent / "adrian.ppn",
    ]
    
    model_exists = any(path.exists() for path in custom_paths)
    
    return {
        "status": "ready" if model_exists else "needs_creation",
        "model_exists": model_exists,
        "built_in_keywords": ["computer", "jarvis", "hey google", "alexa"],
        "custom_model_required": True,
        "offline_capable": True,  # Custom .ppn models work offline
        "instructions": [
            "1. Go to https://console.picovoice.ai (you already have access key!)",
            "2. Click 'Porcupine Wake Word' -> 'Create Wake Word'",
            "3. Enter 'ADRIAN' as the wake word",
            "4. Download the .ppn file",
            f"5. Save it as 'adrian.ppn' in the project 'models' folder: {Path(__file__).parent.parent.parent / 'models'}",
            "6. Once downloaded, the model works completely offline!"
        ],
        "model_locations": [str(path) for path in custom_paths]
    }


def test_hotword_detection() -> bool:
    """
    Test hotword detection system.
    
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Testing hotword detection system...")
    
    if not PORCUPINE_AVAILABLE:
        logger.error("Porcupine not available for testing")
        return False
    
    try:
        # Create test detector
        detector = HotwordDetector(enable_debug=True)
        
        # Try to start
        if not detector.start():
            logger.error("Failed to start detector")
            return False
        
        status = detector.get_status()
        logger.info(f"Detector status: {status}")
        
        detector.stop()
        logger.info("Hotword detection test passed")
        return True
        
    except Exception as e:
        logger.error(f"Hotword detection test failed: {e}")
        return False


# Global instance for easy access
_hotword_detector: Optional[HotwordDetector] = None


def get_hotword_detector() -> Optional[HotwordDetector]:
    """Get global hotword detector instance."""
    return _hotword_detector


def initialize_hotword_detection(
    callback: Optional[Callable[[], None]] = None,
    sensitivity: float = 0.7
) -> HotwordDetector:
    """
    Initialize global hotword detector.
    
    Args:
        callback: Function to call when hotword detected
        sensitivity: Detection sensitivity (0.1-1.0)
        
    Returns:
        Initialized HotwordDetector instance
    """
    global _hotword_detector
    
    if _hotword_detector is None:
        _hotword_detector = HotwordDetector(
            callback=callback,
            sensitivity=sensitivity,
            enable_debug=True
        )
        logger.info("Global hotword detector initialized")
    
    return _hotword_detector
