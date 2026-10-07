# Audio Evaluation Results

> Fill in after running the evaluation tools on your machine.

## Gesture Accuracy (`tools/audio_replay.py` or `tools/audio_eval_sweep.py`)

| Gesture | Hits/Total | Share | Pass (≥80%) |
|---------|------------|-------|-------------|
| Silence | _/_ | _% | _ |
| Speech | _/_ | _% | _ |
| Whisper | _/_ | _% | _ |
| Noise | _/_ | _% | _ |

**Overall:** _

Date: _
Hardware: _ (e.g., Intel i7-1165G7, 16GB RAM, built-in mic)

---

## False-Positive Soak Test (`tools/audio_meter.py`)

| Metric | Result |
|--------|--------|
| Duration | _ minutes |
| Frames processed | _ |
| Average FPS | _ |
| Flags detected | _ |
| Result | PASS / FAIL |

Flags (if any):
- _

Date: _
Hardware: _

---

## Performance

| Metric | Result |
|--------|--------|
| Average FPS (end-to-end) | _ |
| Inference latency per hop | _ ms |
| Memory delta (30 min run) | _ MB |

---

## Notes

- Run `tools/audio_eval_sweep.py manifest.csv` for batch evaluation
- Run `tools/audio_meter.py --list` to list devices, then `tools/audio_meter.py --device N` for live meter
- Run `tools/audio_replay.py recording.wav` for single-file analysis
- Both produce markdown tables for this file