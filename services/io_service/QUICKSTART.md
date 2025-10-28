# ADRIAN Voice Input - Quick Start Guide 🎤

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Picovoice Access Key
Get your free key from: https://console.picovoice.ai/

Add to `.env`:
```bash
PICOVOICE_ACCESS_KEY=your_key_here
```

### 3. Test Components
```bash
python services/io_service/test_service_startup.py
```

### 4. Start the Service
```bash
python -m services.io_service.main
```

That's it! ADRIAN is now listening for "ADRIAN" hotword. 🎉

## Usage

### Basic Flow
1. Say **"ADRIAN"** (wake word)
2. Wait for confirmation log
3. Speak your command: **"open chrome"**
4. Watch the console for transcription

### Conversational Mode
After the hotword, you can continue talking without repeating "ADRIAN":

```
You: "ADRIAN"
[System detects hotword]

You: "What's the weather?"
[System transcribes and processes]

You: "And tomorrow?"  ← No need to say ADRIAN again!
[System still listening in conversation mode]

[After 30s of silence, returns to idle]
```

## Configuration

Edit `.env` or `shared/config.py`:

```bash
# Hotword Detection
HOTWORD_SENSITIVITY=0.7        # 0.1-1.0 (higher = more sensitive)

# Speech-to-Text
WHISPER_MODEL_NAME=base        # tiny|base|small|medium|large
WHISPER_DEVICE=cpu             # cpu or cuda
STT_CONFIDENCE_THRESHOLD=0.5   # 0.0-1.0

# Conversation
CONVERSATION_TIMEOUT_SECONDS=30.0  # Return to idle after silence
MAX_CONTEXT_TURNS=3                # Remember last N exchanges

# VAD
VAD_AGGRESSIVENESS=1          # 0-3 (higher = more aggressive)
```

## Monitoring

### Check Status
```bash
# Health check
curl http://localhost:8001/health

# Conversation status
curl http://localhost:8001/status/conversation

# STT performance
curl http://localhost:8001/status/stt

# Hotword status
curl http://localhost:8001/status/hotword
```

### View Logs
Logs are output to console and stored in `services/io_service/logs/adrian.db`

## Troubleshooting

### "Hotword detection failed"
- Check PICOVOICE_ACCESS_KEY in `.env`
- Get free key from https://console.picovoice.ai/

### "No audio input detected"
- Check microphone permissions
- Run: `python services/io_service/audio_test.py`
- Verify default input device

### "Whisper model download slow"
- First run downloads model (~150MB for "base")
- Be patient, it's a one-time download
- Consider using "tiny" model for faster download

### "Low confidence transcriptions"
- Speak clearly and close to microphone
- Reduce background noise
- Lower `STT_CONFIDENCE_THRESHOLD` in config

### "Hotword not detecting 'ADRIAN'"
- Adjust `HOTWORD_SENSITIVITY` (try 0.6 or 0.8)
- Speak clearly with emphasis on each syllable: "AY-dree-uhn"
- Check microphone input level

## Advanced

### Use Different Whisper Model
Larger models = better accuracy but slower:
```bash
WHISPER_MODEL_NAME=small    # Recommended for better quality
WHISPER_MODEL_NAME=tiny     # Fastest (for testing)
WHISPER_MODEL_NAME=medium   # High quality (slower)
```

### GPU Acceleration
If you have CUDA-capable GPU:
```bash
WHISPER_DEVICE=cuda
```

### Adjust Conversation Timeout
```bash
CONVERSATION_TIMEOUT_SECONDS=60.0  # 1 minute conversations
```

### More Aggressive VAD
Useful in noisy environments:
```bash
VAD_AGGRESSIVENESS=3  # Most aggressive
```

## Testing

### Component Tests
```bash
# Test audio capture
python services/io_service/audio_test.py

# Test hotword detection
python services/io_service/hotword_test.py

# Test VAD
python services/io_service/vad_test.py

# Test STT
python services/io_service/stt_test.py

# Test full startup
python services/io_service/test_service_startup.py
```

### Full Pipeline Test
```bash
python services/io_service/test_pipeline.py
```

### Manual Text Input (Bypass Voice)
```bash
curl -X POST http://localhost:8001/input/text \
  -H "Content-Type: application/json" \
  -d '{"text": "open chrome", "user_id": "test"}'
```

## Next Steps

Once IO Service is running:
1. ✅ Voice input working
2. ⏳ Start Processing Service (NLP)
3. ⏳ Start Execution Service (Actions)
4. ⏳ Start Output Service (TTS)

## Need Help?

- Check `docs/TASK4_VOICE_INPUT_PIPELINE.md` for detailed architecture
- Review logs in `services/io_service/logs/`
- Run startup test: `python services/io_service/test_service_startup.py`

---

**Have fun talking to ADRIAN! 🎉**

