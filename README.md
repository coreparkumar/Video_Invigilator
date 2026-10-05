# Edge Invigilator: Real-Time Body-Gesture Detection on the Edge

A real-time computer vision application that reads a camera stream, estimates human body pose with a perception model, converts the raw landmarks into application-level events (hand raised, looking away, leaving the seat), and logs evidence, all **locally on the device** with no cloud and no network.

Built as a proof of concept for AI-assisted exam invigilation. The focus is the full pipeline around the model: camera input, efficient inference, post-processing, temporal logic, and reliable results, not just model training.

**Stack:** Python, OpenCV, MediaPipe Pose, NumPy | CPU-only inference | Windows and Linux

---

## What it does

- Captures live video from a webcam (auto-discovers the camera index).
- Runs pose estimation per frame and derives scale-invariant geometric features from the landmarks.
- Detects 7 gestures, plus an out-of-frame condition, with a **temporal filter** so a gesture must persist before it is flagged (reduces false positives).
- Calibrates to the user's neutral posture and **adapts slowly** to natural drift, without learning away sustained violations.
- Logs every flag to CSV with a snapshot; shows a live overlay with per-gesture hold timers.

| Gesture | Rule (relative to shoulder width / calibrated baseline) | Flag after |
|---|---|---|
| Hand raised | wrist above nose | 1.0 s |
| Hand at ear | wrist near an ear | 1.5 s |
| Hand covering face | wrist near nose/mouth | 2.0 s |
| Looking left / right | nose offset vs. ear midpoint | 2.5 s |
| Looking down | nose-to-shoulder gap shrinks | 4.0 s |
| Leaning sideways | shoulder-line tilt change > 14 deg | 2.5 s |
| Out of frame | no person detected | 3.0 s |

---

## Pipeline

```
Camera (OpenCV) -> mirror flip -> Pose model (MediaPipe) -> landmarks
   -> feature extraction (normalized by shoulder width)
   -> per-frame classifier (thresholds vs. calibrated baseline)
   -> temporal session (hold timers, cooldown, absence)
   -> evidence log (CSV + one snapshot per flag) + live overlay
```

## Engineering highlights

- **Application layer around a model:** camera handling, pre/post-processing, model-output interpretation, and event logic, the part that turns a perception model into a reliable result.
- **Scale-invariant geometry:** all distances are normalized by shoulder width, so thresholds hold as the user moves closer to or farther from the camera.
- **Temporal filtering and cooldowns:** hold-time thresholds and a repeat-flag cooldown trade a little latency for far fewer false alarms.
- **Adaptive baseline with a safety gate:** the neutral posture follows slow drift (time constant about 20 s) but only while the user is well inside every limit, so a slow creep cannot be absorbed as "normal".
- **Separation of concerns:** config block, pure inference functions (no I/O, time is injected), a state class (`InvigilatorSession`), rendering, and camera code are kept separate, which makes the logic unit-testable with synthetic landmarks and a fake clock.
- **Robust camera startup:** requested index first, then automatic scan of indices 0-3, verifying a real frame can be read.
- **Privacy by design:** processing is in memory; no continuous recording; the only artifacts written are a CSV row and one snapshot per flag; no network calls.
- **Cross-platform:** the detection logic was tested on Linux (Ubuntu, Python 3.12); Windows helper scripts are included.

---

## Setup (Python 3.9-3.12)

```
python -m venv .venv
.venv\Scripts\activate        # Windows   |  source .venv/bin/activate (macOS/Linux)
pip install -r requirements.txt
python gesture_poc.py          # --camera N sets the preferred index; indices 0-3 are auto-scanned
```

On Windows you can instead run `build.bat`, then `run_legacy.bat`. See `BAT_GUIDE.md`.

Versions are pinned on purpose: newer MediaPipe releases dropped the `mp.solutions` API used here (and the newer Tasks API needs a separate model download, which would break the offline design).

## Usage

1. Sit normally, face the screen, press **c** to calibrate your neutral posture.
2. Try the gestures. Each shows a hold timer; it turns red and is **flagged** once it persists past its threshold.
3. Flags are logged to `evidence/events.csv` with a snapshot JPEG.

Keys: `c` calibrate, `s` skeleton on/off, `q` quit.

## Configuration

Hold times live in `GESTURES` and geometric limits in `THRESHOLDS` at the top of `gesture_poc.py`. Adaptive-baseline behavior is controlled by `BASELINE_TAU_S` and `BASELINE_GATE`.

---

## Results

> Fill these in after running the evaluation on your machine. Do not publish numbers you have not measured.

| Metric | Result | Hardware |
|---|---|---|
| Average FPS (end to end) | _TBD_ | _e.g. laptop CPU model_ |
| Inference latency per frame | _TBD_ | |
| Gesture detection (hits out of 10 attempts, per gesture) | _TBD_ | |
| False flags in 2 min of normal sitting/typing | _TBD_ | |

## Testing

- Gesture rules and temporal logic were verified with synthetic landmark inputs (neutral pose, each gesture, near-miss cases, sustained look-away, absence, and baseline drift).
- A step-by-step, test-per-module refactor plan is in `Edge_Invigilator_Vibe_Coding_Plan.md`, with Windows scripts (`steps.bat`, `status.bat`) to run each stage's checks.

---

## Roadmap toward real edge hardware

Planned, not yet implemented:

1. **Optimized inference:** export a detector/pose model to **ONNX**, run with **ONNX Runtime**, then **TensorRT** (FP16/INT8) and compare latency against the current CPU pipeline.
2. **NVIDIA Jetson deployment:** port the pipeline, profile with `tegrastats`, and tune for power and thermals.
3. **Object detection:** add a YOLO or RT-DETR detector for phones and smartwatches, with tracking.
4. **Video pipeline:** GStreamer / DeepStream ingestion for RTSP and multi-camera input.
5. **Packaging and serving:** Docker image and a small REST API for events.
6. **Modular refactor:** split into a tested package (config, features, classifier, session, evidence, camera, pose, overlay).

## Troubleshooting

- **"No camera found" / black window:** the OS is probably blocking camera access for your terminal or IDE.
  - *macOS:* System Settings > Privacy & Security > Camera, enable Terminal / iTerm / VS Code, then restart that app.
  - *Windows:* Settings > Privacy & security > Camera, turn on "Let desktop apps access your camera", and check any hardware camera switch.
  - *Linux:* make sure your user is in the `video` group and `/dev/video0` exists.
- **Camera busy:** close Zoom, Teams, browsers or other apps using the webcam.
- **Wrong camera opens:** use `--camera 1` (or 2, 3).
- **`module 'mediapipe' has no attribute 'solutions'`:** reinstall with `pip install -r requirements.txt`.
- **Low FPS:** set `model_complexity=0` in `gesture_poc.py`, or close other heavy apps.

## Limitations

Single person only; needs head and shoulders in view; head direction is a pose-based estimate (not eye tracking); thresholds are untuned and may vary with lighting, clothing and body type. A flag is a prompt for human review, never proof of misconduct, and any real deployment would need consent, a fairness review and a privacy assessment.
