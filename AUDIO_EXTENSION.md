# Audio Extension: Testing Voice Detection with Video

This document describes how to test the audio detection (whisper/talking) together with video cues using the Edge Invigilator POC.

## Quick Start

### 1. Prerequisites
- Python 3.9-3.12 with virtual environment
- `pip install -r requirements.txt` (includes `sounddevice==0.4.7`, `mediapipe==0.10.14`, `opencv-python`, `numpy`)
- Working microphone and webcam

### 2. Run Synthetic Self-Test (No Hardware Needed)
```bash
python tools/audio_selftest.py
```
Validates the audio pipeline on synthetic signals (ambient, whisper, speech, noise, rustle). Should show **PASS** for all 6 segments.

### 3. Run Live AV Check (Camera + Microphone)
```bash
# Full 9-stage guided test
python tools/av_check.py --camera 0 --device 0 --out evidence

# Audio only (no camera)
python tools/av_check.py --no-video --device 0 --out evidence

# Video only (no microphone)
python tools/av_check.py --no-audio --camera 0 --out evidence

# Run specific stages only (e.g., stages 2, 5, 8)
python tools/av_check.py --stages 2,5,8 --out evidence

# List audio devices
python tools/audio_meter.py --list
```

### 4. Run Batch Evaluation (Offline)
```bash
# Replay a WAV file through the pipeline
python tools/audio_replay.py recording.wav

# Live microphone level meter (calibrates 5s, then shows level bar)
python tools/audio_meter.py --device 0

# Batch evaluation from manifest CSV
python tools/audio_eval_sweep.py manifest.csv
```

## Understanding the 9 Stages

| # | Stage | Duration | What to Do | Expected Result |
|---|-------|----------|------------|-----------------|
| 1 | Device Check | 2s | Verify camera/mic detected | PASS if both open |
| 2 | Audio Calibration | 6s | **Stay completely quiet** | PASS if `calibration_ok` |
| 3 | Video Calibration | 3s | Sit normally, face camera | Visual confirmation |
| 4 | Negative Control | 15s | **Stay silent and still** | No alerts, no audio events |
| 5 | Audio: Whisper | 7s | Whisper continuously | Audio event: "whispering" |
| 6 | Audio: Speech | 7s | Speak normally | Audio event: "talking" |
| 7 | Video: Look Left + Hand | 6s | Look left 3s, raise hand 2s | Video cue "look_left" ≥1.5s |
| 8 | Fused: Whisper + Look Left | 9s | Whisper while looking left | Fused "alert" severity |
| 9 | Burst Control | 3s | Cough twice or rustle paper | No audio event |

## Calibration Status Messages

| Status | Meaning | Action |
|--------|---------|--------|
| `OK` | Calibration successful | Proceed to next stage |
| `NO_SIGNAL` | Mic muted, unplugged, or wrong device | Check connection, try `--device N`, press `n` |
| `CLIPPING` | Input gain too high (peak ≥ 0.98) | Lower OS mic gain, press `n` |
| `TOO_LOUD` | Mean level > -25 dB | Lower gain or move back, press `n` |
| `NOISY` | Std dev > 4 dB during calibration | Quiet environment, close windows, press `n` |

## Keys During Live Test

| Key | Action |
|-----|--------|
| `c` | Calibrate video posture (sit normally, face screen) |
| `n` | Recalibrate audio noise floor (stay quiet first!) |
| `m` | Mute/unmute audio |
| `s` | Toggle skeleton overlay |
| `q` | Quit |

## Output Files

All artifacts written to `--out` folder (default: `evidence/`):

| File | Description |
|------|-------------|
| `events.csv` | Video gesture flags (timestamp, gesture, label, held_seconds, snapshot.jpg) |
| `av_events.csv` | Audio + fused alerts (timestamp, severity, risk, audio_kind, video_cues, reasons, snapshot.jpg) |
| `AV_CHECK_REPORT.md` | Markdown report of all 9 stages with PASS/FAIL/SKIPPED |
| `AV_CHECK_REPORT.md` | Calibration status, per-stage measurements |

## Interpreting Results

### Single Run ≠ Accuracy
A single clean run only proves the pipeline works *for you, in this room, right now*. Real accuracy requires:
- Multiple participants at different distances (0.5m, 1m, 2m, 3m)
- Various noise conditions (silent, HVAC, typing, fan)
- Multiple speakers (different voices, accents)
- Consent and ethics approval (see `docs/AUDIO_ETHICS.md`)

### Ablation Study
Run `tools/eval_ablation.py sessions.csv` to compare:
- Video-only detection
- Audio-only detection  
- Fused (audio + video) detection

Metrics: Precision, Recall, F1, False Alarms/hour.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No camera found" | Close Zoom/Teams/browser; check OS camera permissions |
| "No audio device" | Run `python tools/audio_meter.py --list`; try `--device N` |
| Calibration shows `NO_SIGNAL` | Check mic not muted, correct device selected (`--device N`) |
| Calibration shows `CLIPPING` | Lower mic gain in OS settings; press `n` |
| Calibration shows `NOISY` | Quiet room, close windows, turn off fan, press `n` |
| Whisper not detected | Move closer; note minimum working distance |
| False alerts during negative control | Note in report; may need threshold tuning |

## Privacy & Ethics

- **No audio recorded** - only metadata (labels, dB, timestamps)
- **No transcription** - no speech-to-text, no speaker ID
- **No network** - fully offline
- **Consent required** - see `docs/AUDIO_ETHICS.md` for consent form template
- **Evidence only** - snapshots + CSV logs; human reviews flags

## Files Reference

| File | Purpose |
|------|---------|
| `gesture_poc_av.py` | Main video+audio POC entry point |
| `audio/engine.py` | AudioEngine (orchestrates pipeline, session, fusion) |
| `audio/pipeline.py` | AudioPipeline (buffering, calibration, classification) |
| `audio/baseline.py` | NoiseFloor (calibration, adaptation) |
| `audio/detector.py` | classify_window() - rule-based whisper/speech detection |
| `audio/session.py` | AudioSession (temporal filtering, events) |
| `audio/fusion.py` | Fusion (audio + video → risk-scored alerts) |
| `audio/lips.py` | LipTracker (Face Mesh lip activity cue) |
| `audio/avcheck.py` | AVCheckRunner (pure, 9 stages) |
| `tools/av_check.py` | Live wrapper (camera + mic) |
| `tools/audio_selftest.py` | Synthetic self-test |
| `tools/audio_replay.py` | Offline WAV analysis |
| `tools/audio_meter.py` | Live mic level meter |
| `tools/audio_eval_sweep.py` | Batch evaluation from manifest |
| `tools/eval_ablation.py` | Video vs Audio vs Fused ablation |

## Next Steps

1. Run `python tools/av_check.py` and complete all 9 stages
2. Open `docs/AV_CHECK_REPORT.md` - verify all PASS
3. If any FAIL, note in report and adjust thresholds in `audio/config.py` (marked UNTUNED)
4. Repeat in different conditions (distance, noise, speakers)
5. Run batch evaluation: `python tools/audio_eval_sweep.py manifest.csv`
6. Run ablation: `python tools/eval_ablation.py sessions.csv`

---

*All thresholds in `audio/config.py` are UNTUNED starting points. Calibrate on your own room/mic before claiming accuracy.*