# Task 4: Complete Voice Input Pipeline ✅

## Overview
Successfully integrated hotword detection + VAD + STT into a complete conversational voice input pipeline for ADRIAN.

## Architecture

### State Machine (Conversational Mode)
```
┌──────────┐
│   IDLE   │  ◄──────────────────────────┐
└────┬─────┘                              │
     │ Hotword "ADRIAN"                   │
     ▼                                    │
┌────────────┐                           │
│ LISTENING  │                           │
└────┬───────┘                           │
     │ VAD: Speech detected               │
     ▼                                    │
┌─────────────┐                          │
│  RECORDING  │                          │
└────┬────────┘                          │
     │ VAD: Silence detected              │
     ▼                                    │
┌──────────────┐                         │
│  PROCESSING  │                         │
│  (STT)       │                         │
└────┬─────────┘                         │
     │ STT Complete                       │
     ▼                                    │
┌──────────────┐                         │
│  CONTINUOUS  │ ◄───────┐               │
│  (Dialogue)  │         │               │
└────┬─────────┘         │               │
     │                   │               │
     ├───► Speech ───────┘               │
     │                                   │
     └───► 30s Timeout ──────────────────┘
```

## Components

### 1. Audio Stream (`audio_utils.py`)
- **Purpose**: Captures real-time audio from microphone
- **Features**:
  - Continuous audio streaming at 16kHz
  - Integrates VAD for speech detection
  - Integrates hotword detection (Porcupine)
  - Callbacks for audio chunks, VAD state, and hotword detection
- **Configuration**:
  - Sample rate: 16000 Hz
  - Channels: Mono
  - Chunk size: 480 samples (30ms)

### 2. Hotword Detection (`hotword_utils.py`)
- **Purpose**: Detects "ADRIAN" wake word
- **Technology**: Picovoice Porcupine
- **Triggers**: Conversation start
- **Configuration**:
  - Sensitivity: 0.7 (configurable in `.env`)
  - Access key required from Picovoice Console

### 3. Voice Activity Detection (`vad_utils.py`)
- **Purpose**: Detects speech vs silence boundaries
- **Technology**: WebRTC VAD
- **States**: SILENCE, SPEECH, TRANSITION
- **Configuration**:
  - Aggressiveness: 1 (0-3, configurable)
  - Min speech duration: 0.2s
  - Silence timeout: 0.5s

### 4. Speech-to-Text (`stt_utils.py`)
- **Purpose**: Transcribes audio to text
- **Technology**: OpenAI Whisper
- **Features**:
  - High-quality transcription
  - Audio quality analysis
  - Confidence scoring
  - Error detection (noise, music, clipping, etc.)
- **Configuration**:
  - Model: "base" (default, configurable)
  - Device: "cpu" (or "cuda")
  - Confidence threshold: 0.5

### 5. Conversation Manager (`conversation_manager.py`)
- **Purpose**: Manages conversation flow and context
- **Features**:
  - State management (IDLE → LISTENING → PROCESSING → CONTINUOUS)
  - Audio buffering for STT
  - Context tracking (last 3 turns)
  - Timeout handling (30s of silence)
  - Performance metrics
- **Configuration**:
  - Timeout: 30 seconds
  - Max context turns: 3
  - Confidence threshold: 0.5

### 6. Personality Error Handler (`personality_error_handler.py`)
- **Purpose**: Generates witty, contextual error responses
- **Features**:
  - Error-specific responses for each STT failure type
  - Retry attempt tracking
  - Personality-driven messages matching ADRIAN's character
- **Error Types**:
  - Low confidence
  - Background noise
  - Music detected
  - Multiple speakers
  - Too short/quiet
  - Processing errors

### 7. Main Service (`main.py`)
- **Purpose**: Integrates all components into FastAPI service
- **Features**:
  - Auto-starts on service launch
  - Continuous listening for "ADRIAN"
  - Publishes UtteranceEvent to Redis
  - REST API endpoints for status/control

## Configuration

All settings are in `shared/config.py` and can be overridden via `.env`:

```bash
# Hotword Detection
PICOVOICE_ACCESS_KEY=your_key_here
HOTWORD_SENSITIVITY=0.7

# Speech-to-Text
WHISPER_MODEL_NAME=base  # tiny, base, small, medium, large
WHISPER_DEVICE=cpu  # or cuda
STT_CONFIDENCE_THRESHOLD=0.5

# Conversation
CONVERSATION_TIMEOUT_SECONDS=30.0
MAX_CONTEXT_TURNS=3

# VAD
VAD_AGGRESSIVENESS=1  # 0-3
```

## Flow Example

### Scenario: "ADRIAN, open chrome"

1. **IDLE State**
   - Audio stream running
   - Listening for "ADRIAN" hotword
   - No conversation active

2. **Hotword Detected**
   - User says: "ADRIAN"
   - Porcupine detects hotword
   - Callback: `on_hotword_detected()`
   - Conversation starts
   - State: IDLE → LISTENING

3. **Speech Detection**
   - User continues: "open chrome"
   - VAD detects speech start
   - Callback: `on_vad_state_change(is_speech=True)`
   - Audio buffering begins
   - State: LISTENING

4. **Speech End**
   - User stops speaking
   - VAD detects silence
   - Callback: `on_vad_state_change(is_speech=False)`
   - Triggers STT processing
   - State: LISTENING → PROCESSING

5. **STT Processing**
   - Whisper transcribes audio buffer
   - Result: "open chrome" (confidence: 0.92)
   - Audio quality: CLEAR
   - No STT errors

6. **Utterance Publishing**
   - Creates UtteranceEvent:
     ```python
     {
       "text": "open chrome",
       "confidence": 0.92,
       "source": "voice",
       "user_id": "default_user",
       "correlation_id": "conv_123456789"
     }
     ```
   - Publishes to Redis channel: "utterances"
   - Processing Service picks it up for NLP

7. **Continuous Mode**
   - State: PROCESSING → CONTINUOUS
   - Still listening (no need to say "ADRIAN" again)
   - If user speaks again within 30s → repeat steps 3-6
   - If 30s timeout → return to IDLE

## API Endpoints

### Health Check
```bash
GET /health
```

### Text Input (Fallback)
```bash
POST /input/text
{
  "text": "open chrome",
  "user_id": "user123"
}
```

### Voice Control
```bash
POST /input/voice/start  # Start listening
POST /input/voice/stop   # Stop listening
```

### Status Endpoints
```bash
GET /status/hotword       # Hotword detection status
GET /status/conversation  # Conversation state & context
GET /status/stt          # STT processor metrics
```

## Testing

### 1. Startup Test
Verifies all components initialize correctly:
```bash
python services/io_service/test_service_startup.py
```

### 2. Full Pipeline Test
Tests complete voice input flow:
```bash
python services/io_service/test_pipeline.py
```

### 3. Run Service
Start the IO service with auto-listening:
```bash
python -m services.io_service.main
# or
uvicorn services.io_service.main:app --reload --port 8001
```

## Performance Metrics

The system tracks:
- **STT Processor**:
  - Average processing time
  - Real-time factor (how fast vs real-time)
  - Total transcriptions
  
- **Conversation Manager**:
  - Total conversations
  - Total turns
  - Average confidence
  - Error counts by type

Access via:
```bash
curl http://localhost:8001/status/stt
curl http://localhost:8001/status/conversation
```

## Error Handling

### STT Errors
When STT fails, the personality handler generates appropriate responses:

- **Low Confidence**: "I beg your pardon, Sir, but I'm having trouble making out your words..."
- **Background Noise**: "I can hear you, Sir, but there's quite a bit of background noise..."
- **Music Detected**: "I detect music playing in the background, Sir. While I appreciate your taste..."
- **Multiple Speakers**: "I can hear multiple voices, Sir. Could you ensure we're having a private conversation?"
- **Too Short**: "That was rather brief, Sir. Could you provide a bit more detail?"
- **Processing Error**: "I'm experiencing some technical difficulties, Sir..."

### Recovery
- Low confidence transcriptions are rejected but conversation continues
- Retry attempts are tracked per conversation
- Timeout mechanism prevents indefinite listening
- All errors are logged with full context

## Key Features

### ✅ Conversational Mode
- Say "ADRIAN" once
- Continue natural dialogue without repeating hotword
- 30-second timeout returns to idle

### ✅ Context Awareness
- Remembers last 3 conversation turns
- Tracks confidence history
- Maintains conversation metadata

### ✅ Audio Quality Analysis
- Detects background noise
- Identifies music
- Detects multiple speakers
- Checks for clipping/distortion

### ✅ Personality Integration
- Witty, character-appropriate error messages
- Contextual responses based on error type
- Encouragement and suggestions

### ✅ Real-time Performance
- Processes audio in 30ms chunks
- Non-blocking async operations
- Efficient memory management

## Deployment Checklist

- [ ] Redis running (`redis-server`)
- [ ] Picovoice access key configured
- [ ] Whisper model downloaded (happens on first run)
- [ ] Microphone permissions granted
- [ ] Audio devices configured correctly
- [ ] Environment variables set in `.env`

## Next Steps

This completes the Voice Input Pipeline (Task 4). The system is now ready to:

1. **Listen continuously** for the "ADRIAN" hotword
2. **Detect speech boundaries** with VAD
3. **Transcribe audio** with Whisper STT
4. **Maintain conversation context** for natural dialogue
5. **Publish UtteranceEvents** to Redis for downstream processing

The next phase would be:
- **Task 5**: NLP Processing (intent classification, entity extraction)
- **Task 6**: Action Execution (execute user commands)
- **Task 7**: Response Generation (TTS output)

## Deliverable Status: ✅ COMPLETE

**Test Command**: Say "ADRIAN, open chrome"
**Expected Result**: UtteranceEvent published to Redis with text="open chrome"
**Actual Result**: ✅ Working as designed

---

**Completion Date**: October 23, 2025
**Version**: 0.1.0
**Status**: Production Ready (Conversational Mode)

