# Edge Invigilator POC: body-gesture detection (single laptop + webcam)

Runs fully offline. MediaPipe Pose + OpenCV, CPU only.

## Setup (Python 3.9-3.12)
```
python -m venv .venv
.venv\Scripts\activate        # Windows   |  source .venv/bin/activate (macOS/Linux)
pip install -r requirements.txt
python gesture_poc.py          # --camera N sets the preferred index; indices 0-3 are auto-scanned
```
Versions are pinned on purpose: newer MediaPipe releases dropped the `mp.solutions` API used here.

## Use
1. Sit normally, face the screen, press **c** to calibrate your neutral posture.
2. Try the gestures. Each one shows a hold timer; it turns red and is **flagged** once it persists past its threshold.
3. Flags are logged to `evidence/events.csv` with a snapshot JPEG. Keys: `s` skeleton on/off, `q` quit.

## Gestures detected
| Gesture | Rule (relative to shoulder width / calibrated baseline) | Flag after |
|---|---|---|
| Hand raised | wrist above nose | 1.0 s |
| Hand at ear | wrist near an ear | 1.5 s |
| Hand covering face | wrist near nose/mouth | 2.0 s |
| Looking left / right | nose offset vs. ear midpoint | 2.5 s |
| Looking down | nose-to-shoulder gap shrinks | 4.0 s |
| Leaning sideways | shoulder-line tilt change > 14 deg | 2.5 s |
| Out of frame | no person detected | 3.0 s |

Tune hold times in `GESTURES` and geometric limits in `THRESHOLDS` at the top of `gesture_poc.py`.

**Adaptive baseline:** after calibration, the neutral posture slowly follows you (about 20 s time constant) so chair shifts don't cause false flags. It only adapts while you are well inside every limit (`BASELINE_GATE`), so sustained near-threshold behavior is not learned away. Press **c** to reset it any time.

## Troubleshooting
- **"No camera found" / black window:** the OS is probably blocking camera access for your terminal or IDE.
  - *macOS:* System Settings > Privacy & Security > Camera, enable Terminal / iTerm / VS Code, then restart that app.
  - *Windows:* Settings > Privacy & security > Camera, turn on "Let desktop apps access your camera", and check the camera isn't disabled by a hardware switch.
  - *Linux:* make sure your user is in the `video` group and `/dev/video0` exists.
- **Camera busy:** close Zoom, Teams, browsers or other apps using the webcam.
- **Wrong camera opens:** use `--camera 1` (or 2, 3).
- **`module 'mediapipe' has no attribute 'solutions'`:** you have a newer MediaPipe; reinstall with `pip install -r requirements.txt`.
- **Low FPS:** set `model_complexity=0` in `gesture_poc.py`, or close other heavy apps.

## Limits
Single person only; needs head and shoulders in view; head direction is a pose-based estimate (not eye tracking); thresholds are untuned. A flag is a prompt for human review, not proof of misconduct.
