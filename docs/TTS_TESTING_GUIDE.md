# TTS Testing Guide

## Quick Start: Testing ADRIAN's Voice

### Prerequisites
1. ✅ Microphone volume set to 100% (from previous STT testing)
2. ✅ Speakers/headphones connected and working
3. ✅ Redis running
4. ✅ Python virtual environment activated

---

## Step 1: Install TTS Dependencies

The TTS library should already be in `requirements.txt`. If not installed:

```bash
pip install TTS sounddevice soundfile
```

---

## Step 2: Test TTS Standalone

Test TTS without the full service:

```bash
python services/io_service/test_tts.py --test basic
```

**What to expect:**
- Model will download (~100MB, one-time)
- You'll hear 5 JARVIS-like phrases
- Each phrase will be synthesized and played

**Sample output:**
```
Loading TTS model: tts_models/en/ljspeech/tacotron2-DDC
✅ TTS model loaded successfully
🎙️ Synthesized speech: 'Good morning, Sir. All systems are online.' (1.85s)
🔊 Playing audio...
✅ Playback successful
```

---

## Step 3: Test TTS Queue

Test multiple messages in queue:

```bash
python services/io_service/test_tts.py --test queue
```

**What to expect:**
- 3 messages queued
- Played back-to-back
- No blocking between messages

---

## Step 4: List Available Voice Models

See what other voices are available:

```bash
python services/io_service/test_tts.py --test models
```

**What to expect:**
- List of English TTS models
- Recommendations for JARVIS-like voices

---

## Step 5: Test End-to-End (STT → TTS)

### 5a. Start the IO Service

```bash
python -m services.io_service.main
```

**What to expect:**
```
🚀 Starting IO Service...
✅ Connected to Redis
✅ Whisper model loaded successfully
✅ TTS model loaded successfully
✅ TTS response handler started
🎉 IO Service startup complete - ready for voice input!
```

### 5b. Run the End-to-End Test

In a **new terminal**:

```bash
python services/io_service/test_end_to_end.py
```

**Menu options:**
1. **TTS Response (basic)** - Test TTS with simple phrases
2. **Full Conversation** - Simulate a weather query conversation
3. **Error Responses** - Test personality-driven error messages
4. **All Tests** - Run all tests sequentially

**What to expect:**
- Test script publishes `ResponseText` events to Redis
- IO Service picks them up
- ADRIAN speaks the responses through your speakers

---

## Step 6: Test Real Voice Conversation

### Full Voice Pipeline Test

1. **Start IO Service** (if not already running):
   ```bash
   python -m services.io_service.main
   ```

2. **Say "ADRIAN"** to your microphone
   - You should see: `🎯 ADRIAN hotword detected!`

3. **Say a command** (e.g., "Hello ADRIAN")
   - You should see: `🎤 Transcribed: 'Hello ADRIAN'`
   - Utterance published to Redis

4. **Wait for response** (requires Processing Service)
   - Processing Service would generate a response
   - IO Service would speak it via TTS

---

## Troubleshooting

### Issue: "No audio output"

**Check:**
1. Speaker volume is up
2. Correct audio output device selected
3. Run system audio test:
   ```python
   import sounddevice as sd
   sd.play([0.5] * 44100, 44100)  # 1 second beep
   sd.wait()
   ```

### Issue: "Model download fails"

**Solution:**
- Check internet connection
- Check disk space (~500MB needed)
- Manually download from: https://github.com/coqui-ai/TTS
- Place in `~/.local/share/tts/`

### Issue: "TTS too slow"

**Solution:**
1. Use GPU if available:
   ```python
   # In .env or config
   TTS_DEVICE=cuda
   ```

2. Or switch to faster model:
   ```python
   TTS_MODEL_NAME=tts_models/en/ljspeech/glow-tts
   ```

### Issue: "Voice sounds robotic"

**Solution:**
1. Adjust speed in config:
   ```python
   TTS_SPEED=1.0  # Try different values: 0.9, 1.0, 1.1, 1.2
   ```

2. Try different model:
   ```python
   TTS_MODEL_NAME=tts_models/en/vctk/vits
   ```

---

## Voice Customization

### Adjusting Speed

In `shared/config.py` or `.env`:
```python
tts_speed: float = 1.1  # 0.5-2.0
```

- **0.9-1.0**: Slower, more deliberate (butler-like)
- **1.1-1.2**: Faster, more efficient (JARVIS-like) ✅
- **1.3-1.5**: Very fast (urgent situations)

### Trying Different Models

Edit `shared/config.py`:
```python
tts_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
```

**Options:**
- `tts_models/en/ljspeech/tacotron2-DDC` - Current (good balance)
- `tts_models/en/ljspeech/glow-tts` - Faster
- `tts_models/en/vctk/vits` - Multi-speaker (can select voice)

---

## Performance Benchmarks

### Expected Performance (CPU - Intel i7)
- **First synthesis**: 3-5 seconds (model loading)
- **Subsequent synthesis**: 1-2 seconds
- **Playback**: Real-time (no delay)

### With GPU (NVIDIA RTX)
- **First synthesis**: 2-3 seconds
- **Subsequent synthesis**: 0.5-1 second
- **Playback**: Real-time

---

## Next Steps

After TTS is working:

1. **✅ Voice Input (STT)** - Working
2. **✅ Voice Output (TTS)** - Working
3. **⏳ Processing Service** - Next task
4. **⏳ Connect STT → Processing → TTS** - Final integration

---

## Success Criteria

TTS is working correctly when:

- [x] Model loads without errors
- [x] Test phrases play through speakers
- [x] Voice sounds natural (not too robotic)
- [x] Speed is appropriate (JARVIS-like)
- [x] Queue handles multiple messages
- [x] No audio glitches or cutoffs
- [x] Performance is acceptable (<2s synthesis)

---

## Support

If you encounter issues:

1. Check logs for specific errors
2. Verify all dependencies installed
3. Test audio output device
4. Try different TTS model
5. Check Redis connection

**Common fixes:**
- Restart IO Service
- Clear TTS cache: `rm -rf ~/.local/share/tts/`
- Reinstall TTS: `pip install --upgrade TTS`

---

## Summary

You now have a complete voice I/O system:

**Input**: Microphone → Hotword → VAD → STT → Text
**Output**: Text → TTS → Speakers

Ready to connect to the Processing Service for full conversational AI! 🎤🤖🔊

