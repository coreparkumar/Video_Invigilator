# Edge Invigilator POC: Real-Time Body-Gesture Detection on the Edge

A proof of concept that shows a laptop webcam can be analyzed locally, in real time, to spot body gestures that may indicate exam misconduct, and log them for human review. No cloud, no network, no continuous recording.

**Stack:** Python 3.9-3.12, OpenCV, MediaPipe 0.10.14 Pose, NumPy | CPU-only | Windows/Linux

---

## What it does

- Captures live video from a webcam (auto-discovers camera index 0-3)
- Runs MediaPipe Pose estimation per frame
- Derives scale-invariant geometric features from landmarks (normalized by shoulder width)
- Detects 7 gestures + out-of-frame with temporal filtering (gesture must persist before flagging)
- Calibrates to user's neutral posture (`c` key) and adapts slowly to natural drift without learning away violations
- Logs every flag to `evidence/events.csv` with one snapshot JPEG per flag
- Shows live overlay: skeleton, per-gesture hold timers, FPS, calibration status, flag banners

### Gestures

| Gesture | Label | Flag after |
|---------|-------|------------|
| `hand_raised` | Hand raised | 1.0 s |
| `hand_to_ear` | Hand at ear (phone/earpiece?) | 1.5 s |
| `hand_to_face` | Hand covering face/mouth | 2.0 s |
| `look_left` | Looking left | 2.5 s |
| `look_right` | Looking right | 2.5 s |
| `look_down` | Looking down | 4.0 s |
| `leaning` | Leaning sideways | 2.5 s |
| `out_of_frame` | Out of frame / left seat | 3.0 s |

---
## Demo
<img width="712" height="567" alt="image" src="https://github.com/user-attachments/assets/ebe254a0-9db8-4a59-804c-92f0378574bf" />


## Quick Start

```bash
# 1. Create venv and install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt

# 2. Run video-only POC
python -m invigilator            # or: .venv\Scripts\python.exe -m invigilator

# 3. Run video + audio POC (new!)
python gesture_poc_av.py         # or: .venv\Scripts\python.exe gesture_poc_av.py

# Optional args (both apps):
python -m invigilator --camera 1    # preferred camera index
python -m invigilator --video file.mp4  # replay from video file
python -m invigilator --out my_evidence  # custom output folder
python -m invigilator --no-snapshots     # disable snapshot saving

# Audio-specific args (gesture_poc_av.py only):
python gesture_poc_av.py --audio-device 1  # preferred mic index
python gesture_poc_av.py --no-audio        # disable audio
python gesture_poc_av.py --wav file.wav    # replay from audio file
```

### Keys
- `c` — Calibrate neutral posture (sit normally, face screen, press)
- `n` — Recalibrate audio noise floor (stay quiet, press `n`) **[Audio]**
- `m` — Mute/unmute audio **[Audio]**
- `s` — Toggle skeleton overlay
- `q` — Quit

---

## Project Structure

```
edge_invigilator/
├── invigilator/
│   ├── __init__.py
│   ├── __main__.py          # entry point
│   ├── app.py               # main loop wiring
│   ├── config.py            # all thresholds & constants
│   ├── features.py          # landmarks -> scale-invariant measurements
│   ├── classifier.py        # measurements -> active gestures (per frame)
│   ├── session.py           # hold timers, cooldowns, flags
│   ├── baseline.py          # calibration + adaptive baseline
│   ├── evidence.py          # CSV + snapshot logging
│   ├── camera.py            # camera discovery + FrameSource protocol
│   ├── pose.py              # MediaPipe wrapper
│   └── overlay.py           # OpenCV drawing
├── tests/
│   ├── conftest.py          # synthetic landmarks, FakeClock
│   ├── test_characterization.py  # legacy behavior capture
│   ├── test_config.py
│   ├── test_features.py
│   ├── test_classifier.py
│   ├── test_session.py
│   ├── test_baseline.py
│   ├── test_evidence.py
│   ├── test_camera.py
│   ├── test_pose.py
│   ├── test_overlay.py
│   └── test_app.py
├── tools/
│   ├── eval_gestures.py     # guided 10-attempt accuracy test
│   └── soak_test.py         # false-positive soak test
├── docs/
│   └── EVAL.md              # evaluation results (generated)
├── legacy/
│   └── gesture_poc.py       # frozen baseline (v0.2)
├── requirements.txt
└── README.md
```

**Dependency direction:** `config` → `features` → `classifier` → `session` → `app`; `evidence`, `camera`, `pose`, `overlay` are leaves used only by `app`.

---

## Configuration

All tunable values in `invigilator/config.py`:

- `GESTURES` — label + hold_seconds per gesture
- `THRESHOLDS` — geometric limits (normalized by shoulder width)
- `DEFAULT_BASELINE` — yaw/neck/tilt before calibration
- `COOLDOWN_S` — 5.0 s minimum between repeat flags
- `VIS_MIN` — 0.5 landmark visibility threshold
- `ABSENT_GRACE_S` — 0.5 s before out-of-frame starts counting
- `BASELINE_TAU_S` — 20.0 s adaptation time constant
- `BASELINE_GATE` — 0.5 (only adapt while within 50% of every threshold)

Run validation: `python -m invigilator.config`

---

## Testing

```bash
# All tests
.venv\Scripts\python.exe -m pytest tests/ -q

# Per-module (using helper scripts)
steps.bat    # pick module 0-12
status.bat   # automated pass/fail table
test_all.bat # full suite
```

### Test Coverage (125 tests)
- **Characterization (12):** Legacy behavior frozen — neutral, each gesture, near-miss, scale invariance
- **Config (10):** Values match legacy; validation rejects bad configs
- **Features (10):** Scale invariance, tilt abs(dx), zero shoulder width, near_neutral gate
- **Classifier (21):** Each gesture + just-under/just-over thresholds, calibrated baseline shifts, priority rules
- **Session (16):** Flag at hold time, flicker resets, cooldown blocks then allows, absence grace+hold, multiple gestures
- **Baseline (14):** Calibration, reset, adapt with gate, small drift absorbed, large look-away not absorbed, creep contained
- **Evidence (9):** Header once, one row+snapshot per flag, no extra files, context manager
- **Camera (13):** FakeSource, auto-discovery order, video file, all-fail error
- **Pose (9):** Black frame -> None, close idempotent, context manager, helpful error on wrong MediaPipe
- **Overlay (8):** Shape preserved, session_view not mutated, all gestures, draw_skeleton edge cases
- **App (3):** End-to-end with FakeSource+stub pose → exactly 1 CSV row + 1 snapshot

---

## Evaluation

### Guided Accuracy Test (10 attempts per gesture)
```bash
.venv\Scripts\python.exe tools/eval_gestures.py eval --camera 0 --out evidence
```
Prompts you to perform each gesture 10 times, holds for 3s. Writes `evidence/EVAL.md`.

**PRD Target:** ≥ 8/10 per gesture

### False-Positive Soak Test (2 min default)
```bash
.venv\Scripts\python.exe tools/soak_test.py --camera 0 --minutes 2 --out evidence
```
Sit and type normally. Reports FPS, flags detected, memory. Writes `evidence/SOAK.md`.

**PRD Target:** 0 flags in 2 min; ≥ 15 FPS; 30-min run stable

---

## Audio Extension: Voice + Video Testing

The `gesture_poc_av.py` adds real-time audio detection (whisper/talking) fused with video cues.

### Quick Audio Test (No Hardware)
```bash
# Synthetic self-test (validates pipeline on synthetic signals)
python tools/audio_selftest.py
```

### Live Audio + Video Test (9 Stages)
```bash
# Full guided test (camera + microphone)
python tools/av_check.py --camera 0 --device 0 --out evidence

# Audio only
python tools/av_check.py --no-video --device 0 --out evidence

# Video only
python tools/av_check.py --no-audio --camera 0 --out evidence

# Specific stages only
python tools/av_check.py --stages 2,5,8 --out evidence
```

### 9-Stage Guided Test
| # | Stage | Duration | Action | Expected |
|---|-------|----------|--------|----------|
| 1 | Device Check | 2s | Verify camera/mic | Both open |
| 2 | Audio Calibration | 6s | **Stay quiet** | `calibration_ok` |
| 3 | Video Calibration | 3s | Sit normally | Visual check |
| 4 | Negative Control | 15s | Silent & still | No alerts/events |
| 5 | Audio: Whisper | 7s | Whisper continuously | Audio event: "whispering" |
| 6 | Audio: Speech | 7s | Speak normally | Audio event: "talking" |
| 7 | Video: Look Left + Hand | 6s | Look left 3s, raise hand 2s | Video cue ≥1.5s |
| 8 | Fused: Whisper + Look Left | 9s | Whisper while looking left | Fused "alert" |
| 9 | Burst Control | 3s | Cough/rustle paper | No audio event |

### Audio Calibration Status
| Status | Meaning | Action |
|--------|---------|--------|
| `OK` | Calibration successful | Proceed |
| `NO_SIGNAL` | Mic muted/unplugged/wrong device | Check connection, try `--device N`, press `n` |
| `CLIPPING` | Gain too high (peak ≥ 0.98) | Lower OS mic gain, press `n` |
| `TOO_LOUD` | Level > -25 dB | Lower gain or move back, press `n` |
| `NOISY` | Std dev > 4 dB during cal | Quiet room, close windows, press `n` |

### Audio Keys
| Key | Action |
|-----|--------|
| `c` | Calibrate video posture |
| `n` | Recalibrate audio noise floor (stay quiet first!) |
| `m` | Mute/unmute audio |
| `s` | Toggle skeleton overlay |
| `q` | Quit |

### Audio Evaluation Tools
```bash
# Live microphone meter (calibrates 5s, then level bar)
python tools/audio_meter.py --device 0

# Offline WAV analysis (timeline + summary)
python tools/audio_replay.py recording.wav

# Batch evaluation from manifest CSV (requires consent_ok=yes)
python tools/audio_eval_sweep.py manifest.csv
# Writes docs/AUDIO_EVAL.md with detection rates by distance/noise

# Ablation study: video-only vs audio-only vs fused
python tools/eval_ablation.py sessions.csv
# Writes docs/ABLATION.md with precision/recall/F1
```

**PRD Targets (Audio):**
- ≥ 8/10 per gesture (whisper, speech, cough, rustle)
- 0 flags in 2-min quiet soak
- ≥ 15 FPS end-to-end

See `AUDIO_EXTENSION.md` for complete testing guide.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No camera found" | OS blocking camera access. **Windows:** Settings > Privacy > Camera > enable for Terminal/VS Code. **macOS:** System Settings > Privacy > Camera > enable Terminal/iTerm. **Linux:** add user to `video` group. |
| Camera busy | Close Zoom, Teams, browsers |
| Wrong camera opens | Use `--camera 1` (or 2, 3) |
| `module 'mediapipe' has no attribute 'solutions'` | `pip install -r requirements.txt` (needs MediaPipe 0.10.14) |
| Low FPS | Close heavy apps; or edit `PoseEstimator(model_complexity=0)` in `app.py` |

---

## Privacy & Ethics

- Processing is **local**; only `events.csv` and flag snapshots in `evidence/` are written
- **No network calls**, no continuous video recording
- Users must consent to monitoring; a flag is a **prompt for human review**, never proof
- Known limits: single person, head+shoulders visible, pose-based head direction (not eye tracking), untuned thresholds vary with lighting/clothing/body type
- Fairness review required before any real deployment

---

## Roadmap (Post-POC)

1. **ONNX/TensorRT export** for optimized inference on Jetson-class hardware
2. **Object detection** (phones, watches) with YOLO/RT-DETR + tracking
3. **Multi-camera / RTSP** via GStreamer/DeepStream
4. **Docker + REST API** for event serving
5. **Face landmarks** for better gaze estimation

---

## License

MIT — see LICENSE file.