# ADRIAN Voice Pipeline Guide

This guide covers end-to-end setup, testing, optimisation, and troubleshooting for the ADRIAN voice pipeline. It reflects the Phase 2 deliverable: a production-ready, low-latency voice experience with rich controls.

---

## 1. Setup Checklist

1. **Dependencies**
   - Install/update the project requirements after pulling these changes:
     ```bash
     pip install -r requirements.txt
     ```
   - New dependencies: `pystray` and `Pillow` (for the tray indicator).

2. **Configure Audio Devices** (`shared/config.py` or `.env`)
   | Setting | Purpose |
   |---------|---------|
   | `microphone_device_index` / `microphone_device_name` | Select input mic (index wins over name). |
   | `speaker_device_index` / `speaker_device_name` | Select playback device for TTS. |
   | `activation_beep_*`, `error_beep_*` | Tune activation/error feedback tones. |

   Use `GET http://127.0.0.1:8001/audio/devices` to list available devices before picking an index/name.

3. **Voice Settings**
   - Default voice: `p234` (VCTK male British).
   - Runtime adjustments via Output Service API (port `8006`):
     - `POST /tts/voice` `{ "speaker": "p227" }`
     - `GET /tts/voices`
     - `POST /tts/output-device` (`index` or `name`)
     - `POST /tts/volume` `{ "volume": 0.8 }`

4. **Service Startup (Hybrid Windows/WSL)**
   - Windows PowerShell: `.start_adrian_windows.ps1` (IO service: mic access)
   - WSL: `./start_adrian_wsl.sh` (Processing + Output services)

---

## 2. Integration Test Procedure

1. **Start all services** (as above) and confirm:
   - IO Service console prints _“IO Service ready - Say 'ADRIAN' to start!”_
   - Output Service announces readiness and tray icon turns blue (idle).

2. **Scenario Test** – say: “**ADRIAN, what time is it?**”
   - Observe pipeline order: hotword → speech capture → Processing → spoken response.
   - Listening beep plays immediately after hotword; tray icon turns yellow (speaking) during playback.

3. **Measure Latency**
   - Call `GET http://127.0.0.1:8001/status/pipeline`
   - Ensure `hotword_to_response` average < 2000 ms (target). Investigate alerts if shown.

4. **Queue & Interruption Test**
   - Trigger back-to-back requests via `/speak` endpoint.
   - While ADRIAN speaks, say “ADRIAN” to interrupt; playback stops and microphone resumes immediately.

5. **Background Noise / Unclear Speech**
   - Speak with intentional noise or mumbling.
   - IO service should respond with an apologetic prompt; after `conversation_retry_limit` attempts (default `2`) ADRIAN gracefully exits listening mode.

---

## 3. Optimisation & Monitoring

- **Latency Tracking**
  - `GET /status/pipeline` exposes averages plus recent event history.
  - Alerts trigger when the hotword→response average exceeds the configured `pipeline_latency_target_ms`.

- **Hotword Sensitivity**
  - Adjust in config: `hotword_sensitivity` (default `0.7`).
  - Use `/audio/devices` to confirm mic level and try alternative hardware if needed.

- **VAD & Confidence**
  - `vad_aggressiveness` (`0`–`3`) and `stt_confidence_threshold` tune speech detection vs. background rejection.
  - `conversation_retry_limit` controls the number of “I didn’t catch that” prompts before resetting.

- **Output Service Tray Indicator**
  - Blue: idle; Yellow: speaking; Green (reserved for future “listening” state); Red: TTS error.
  - Disable via `tray_indicator_enabled = False` if the environment does not support system tray icons.

---

## 4. Troubleshooting

| Symptom | Suggested Fix |
|---------|---------------|
| Hotword never triggers | Verify mic device selection, raise `hotword_sensitivity`, confirm Porcupine access key (if using the custom model). |
| ADRIAN hears itself (“echo loop”) | Ensure IO service is running on Windows, confirm the Output service publishes `tts_events`, check that WSL playback uses Windows speakers (PowerShell playback fallback). |
| Voice sounds too slow/high | Adjust `tts_speed` (1.0–1.2 recommended) and `tts_pitch_shift` (-3.0 to 0.0). |
| No audio output | Set `speaker_device_index` explicitly, or call `POST /tts/output-device` with the correct device. |
| Tray icon missing | Ensure Windows Explorer is running; disable the indicator (`tray_indicator_enabled=False`) if not supported. |
| Latency > 2 s | Inspect `/status/pipeline` history, check CPU/GPU load, consider switching Whisper model (`small` or `tiny`) and ensure no background tasks overload CPU. |

---

## 5. Voice Model Selection

Recommended VCTK voices (male British):

| Speaker | Region | Tone |
|---------|--------|------|
| `p234` | Manchester | Default (authoritative) |
| `p334` | London | Rich, smooth |
| `p227` | Southern | Warm |
| `p256` | Scottish | Crisp |
| `p230` | Northern | Neutral |

Use `python services/io_service/test_all_voices.py --speakers p234 p334 p227 ...` to audition candidates quickly.

For experimental cloning (XTTS), set `tts_use_voice_cloning=True` and provide `speaker_wav`, but note the additional model requirements and load time.

---

## 6. Quick Reference API Table

| Service | Endpoint | Purpose |
|---------|----------|---------|
| IO (`8001`) | `GET /status/pipeline` | Latency metrics + event history |
| IO (`8001`) | `GET /audio/devices` | Enumerate input/output devices |
| Output (`8006`) | `POST /tts/voice` | Change speaker voice |
| Output (`8006`) | `GET /tts/voices` | List available voices |
| Output (`8006`) | `POST /tts/output-device` | Select playback device |
| Output (`8006`) | `POST /tts/volume` | Adjust playback volume |
| Output (`8006`) | `POST /tts/stop` | Interrupt speaking |

---

## 7. Recommended Workflow

1. Calibrate audio devices (`/audio/devices`, config fields).
2. Run integration test prompt (“ADRIAN, what time is it?”).
3. Confirm latency target via `/status/pipeline`.
4. Tune sensitivity/volume/voice as desired through Output Service APIs.
5. Document findings or custom configs inside `.env` for persistence.

With these steps ADRIAN now delivers a responsive, Jarvis-like voice experience with live metrics, device configurability, and resilient error handling.


