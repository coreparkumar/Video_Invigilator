# Evaluation Results

> Fill in after running the evaluation tools on your machine.

## Gesture Accuracy (`tools/eval_gestures.py eval`)

| Gesture | Hits/10 | Pass (≥8) |
|---------|---------|-----------|
| Hand raised | _/10 | _ |
| Hand at ear (phone/earpiece?) | _/10 | _ |
| Hand covering face/mouth | _/10 | _ |
| Looking left | _/10 | _ |
| Looking right | _/10 | _ |
| Looking down | _/10 | _ |
| Leaning sideways | _/10 | _ |

**Overall:** _

Date: _
Hardware: _ (e.g., Intel i7-1165G7, 16GB RAM)

---

## False-Positive Soak Test (`tools/soak_test.py`)

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
| Inference latency per frame | _ ms |
| Memory delta (30 min run) | _ MB |

---

## Notes

- Run `tools/eval_gestures.py eval --camera 0 --out evidence` for accuracy test
- Run `tools/soak_test.py --camera 0 --minutes 2 --out evidence` for soak test
- Both write markdown here (`docs/EVAL.md`)