# Edge Invigilator POC: Vibe-Coding Implementation Plan

**Basis:** Simplified PRD v0.2, `gesture_poc.py` (monolith), `README.md`, `requirements.txt` **Goal:** Rebuild the working POC as small, testable modules, one at a time, with a human gate between every step. **Scope reminder:** one laptop, one webcam, one person, offline, CPU only, 7 gestures. Anything else is out of scope until you say otherwise.

---

## 0. How to Use This Plan

1. Open a fresh AI coding session (or chat) and paste the **Master Rules Prompt** (Section 1) once.
2. For each module (M0 to M12), paste its **Prompt**. The AI builds only that module, runs the checks, and prints a **Checkpoint Report**.
3. You review. If it passes, type the module's **CONTINUE trigger**. If anything looks wrong, type a **STOP command**.
4. Never skip a module. Never paste two module prompts at once.

### Control Commands (you type these)

| Command | Effect |
| --- | --- |
| `CONTINUE Mx` | Approves the previous checkpoint and starts module Mx. |
| `STOP` | AI stops immediately, makes no further edits, prints current state. |
| `STATUS` | AI prints: current module, files changed, test results, open issues. |
| `REVERT Mx` | AI undoes all changes from module Mx (git revert/restore) and confirms. |
| `RESCOPE: <text>` | AI re-states the module goal with your change and waits for `CONTINUE`. |

### Auto-Halt Conditions (the AI must stop by itself and print `HALT: <reason>`)

- A test fails and a second fix attempt also fails.
- The change needs a new dependency, a network call, or a file outside the module's allowed list.
- The change exceeds the module's line budget by more than 50%.
- The request is ambiguous or conflicts with the PRD or these rules.
- Any code would write video to disk (other than a single flag snapshot) or send data off the machine.

---

## 1. Master Rules Prompt (paste once at the start)

```text
You are my pair-programmer on "Edge Invigilator POC": local, offline body-gesture
detection from a laptop webcam (Python 3.9-3.12, OpenCV, MediaPipe 0.10.14 Pose).

WORKING RULES
1. One module at a time. Build ONLY the module I name. Do not touch other modules.
2. Small steps: write the tests first (or alongside), then the code, then run both.
3. Pure logic stays pure: no camera, no UI, no file I/O, no time.time() inside
   classifiers or session logic. Time is always passed in as `now`.
4. No new dependencies beyond requirements.txt (mediapipe==0.10.14,
   opencv-python==4.10.0.84, numpy<2) plus pytest for tests. Ask first otherwise.
5. Privacy: no network calls, no continuous video recording. The only artifacts
   written are the events.csv row and ONE snapshot per flag.
6. No magic numbers: every threshold lives in config.py.
7. Keep functions short, typed, and docstringed in one line. No clever abstractions.
8. After each module, print a CHECKPOINT REPORT:
   - What was built (files + line counts)
   - Tests run and results (paste output summary)
   - Manual check I should do (exact command)
   - Risks / things you were unsure about
   Then STOP and wait for my command.
9. Obey my control commands: CONTINUE Mx, STOP, STATUS, REVERT Mx, RESCOPE.
10. Print "HALT: <reason>" and stop if any auto-halt condition occurs
    (repeated test failure, new dependency, out-of-scope file, ambiguity,
    privacy violation, >50% over the line budget).

Reply "READY" and nothing else.
```

**Continue trigger:** AI replies `READY` → you type `CONTINUE M0`.

---

## 2. Target Structure

```text
edge_invigilator/
  invigilator/
    config.py        # gestures, thresholds, constants (M1)
    features.py      # landmarks -> measurements (M2)
    classifier.py    # measurements -> active gestures (M3)
    session.py       # hold timers, cooldown, absence, flags (M4)
    baseline.py      # calibration + adaptive baseline (M5)
    evidence.py      # events.csv + snapshot (M6)
    camera.py        # camera discovery + frame source (M7)
    pose.py          # MediaPipe wrapper (M8)
    overlay.py       # OpenCV drawing (M9)
    app.py           # main loop + CLI (M10)
  tests/
    conftest.py      # synthetic landmark builders, fake clock
    test_*.py
  tools/
    soak_test.py     # false-positive soak (M11)
  README.md  requirements.txt
```

**Dependency direction (never reverse it):** `config` \<- `features` \<- `classifier` \<- `session` \<- `app`; `evidence`, `camera`, `pose`, `overlay` are leaves used only by `app`.

---

## 3. Modules

### M0: Freeze the Baseline and Scaffold

**Goal:** Protect the working POC, create the project skeleton, and capture current behavior as tests. **Allowed files:** new package folders, `tests/conftest.py`, `tests/test_characterization.py`. Existing `gesture_poc.py` is read-only. **Line budget:** \~120.

```text
MODULE M0: Freeze baseline and scaffold.
1. Confirm gesture_poc.py is committed/tagged (git tag poc-v0.2) or copy it to
   legacy/gesture_poc.py. Do not modify it.
2. Create the folder structure from the plan with empty __init__.py files.
3. In tests/conftest.py build helpers: make_landmarks(**overrides) returning a
   33-point list of SimpleNamespace(x,y,visibility) for a neutral seated person,
   and a FakeClock class with advance(seconds).
4. In tests/test_characterization.py import classify/measures from the legacy
   file and assert current behavior: neutral -> empty set; wrist above nose ->
   hand_raised; nose shifted right -> look_right; shoulder dropped -> leaning;
   nose lowered -> look_down.
5. Run pytest. Report. Then STOP.
```

**Checks:** `pytest -q` passes; `git status` shows `gesture_poc.py` unchanged. **STOP if:** the legacy tests fail (the baseline is not what we think it is). **CONTINUE trigger:** all characterization tests green → `CONTINUE M1`.

---

### M1: Configuration

**Goal:** Move every tunable value into one validated config module. **Allowed files:** `invigilator/config.py`, `tests/test_config.py`. **Line budget:** \~100.

```text
MODULE M1: config.py.
Create frozen dataclasses (or plain dicts + a validate() function) holding:
GESTURES (key -> label, hold_seconds), THRESHOLDS (hand_raised_above_nose,
hand_to_face_dist, hand_to_ear_dist, yaw_shift, neck_ratio, tilt_deg),
DEFAULT_BASELINE, COOLDOWN_S, VIS_MIN, ABSENT_GRACE_S, BASELINE_TAU_S,
BASELINE_GATE, CAMERA_SCAN, and landmark index constants. Values must match
gesture_poc.py exactly.
Add validate(): every hold time > 0, ratios within 0..1 where applicable,
GESTURES has all 8 keys. Tests: values equal the legacy file's values;
validate() rejects a bad value. Run pytest, report, STOP.
```

**Checks:** a test compares each constant to the legacy value. **STOP if:** any value differs from legacy without your approval. **CONTINUE trigger:** green tests and no value drift → `CONTINUE M2`.

---

### M2: Feature Extraction (pure)

**Goal:** Convert landmarks into scale-invariant measurements. **Allowed files:** `invigilator/features.py`, `tests/test_features.py`. **Line budget:** \~70.

```text
MODULE M2: features.py.
Port measures(lm) -> dict(sh_w, yaw, neck, tilt) and dist(a, b). Pure functions,
type hints, no I/O. Use indices and constants from config.py.
Tests: (a) neutral pose gives tilt ~0, yaw ~0; (b) doubling all coordinates'
distance from the origin scale does NOT change yaw, neck, tilt (scale invariance
via shoulder width); (c) tilt is unchanged if the left/right labels are swapped
(abs on dx); (d) degenerate input (zero shoulder width) does not divide by zero.
Run pytest, report, STOP.
```

**STOP if:** the scale-invariance test fails (normalization is wrong). **CONTINUE trigger:** → `CONTINUE M3`.

---

### M3: Gesture Classifier (pure)

**Goal:** Map measurements + baseline to the set of active gestures for one frame. **Allowed files:** `invigilator/classifier.py`, `tests/test_classifier.py`. **Line budget:** \~90.

```text
MODULE M3: classifier.py.
Port classify(lm, base, th) -> set[str] covering: hand_raised, hand_to_face,
hand_to_ear, look_left, look_right, look_down, leaning. (out_of_frame belongs
to session, not here.)
Rules to preserve: face check takes priority over ear; ear requires the wrist
not above the nose; wrists below VIS_MIN visibility are ignored.
Table-driven pytest: one case per gesture (positive), one near-miss per gesture
(just under threshold -> NOT flagged), one neutral case, and one two-gestures-at-
once case. Also run the M0 characterization tests against the new classifier.
Report, STOP.
```

**Checks:** each threshold has a "just under / just over" test pair. **STOP if:** any near-miss test is flaky, since the threshold semantics need your decision. **CONTINUE trigger:** → `CONTINUE M4`.

---

### M4: Temporal Session Logic (no I/O)

**Goal:** Hold timers, cooldowns, out-of-frame detection, flag events. This is the false-positive filter. **Allowed files:** `invigilator/session.py`, `tests/test_session.py`. **Line budget:** \~110.

```text
MODULE M4: session.py (pure; time passed as `now`; NO file or camera access).
class Session with update(landmarks_or_None, now) -> list[FlagEvent], where
FlagEvent = (gesture_key, label, held_seconds). Track: since{}, last_flag{},
absent_since, active set, banners data. Behavior:
- a gesture flags only after it has persisted for its hold time;
- it resets if it disappears for even one frame;
- after flagging, the same gesture cannot flag again within COOLDOWN_S;
- landmarks None for > ABSENT_GRACE_S makes out_of_frame active (hold 3.0 s);
- low-visibility head/shoulders are treated as "no reliable pose" (no classify).
Tests with FakeClock: flag exactly at hold time (not before); flicker resets the
timer; cooldown blocks the repeat then allows it after 5 s; absence flags at
~3.5 s total; start timestamps like 1000.0 (not 0) to catch cooldown-at-zero bugs.
Report, STOP.
```

**STOP if:** any test needs `time.sleep()`, since time must stay injected. **CONTINUE trigger:** → `CONTINUE M5`.

---

### M5: Calibration and Adaptive Baseline

**Goal:** User-triggered neutral baseline that slowly follows natural posture drift without learning away real violations. **Allowed files:** `invigilator/baseline.py`, small hook in `session.py`, `tests/test_baseline.py`. **Line budget:** \~80.

```text
MODULE M5: baseline.py.
class Baseline: calibrate(measures) stores yaw/neck/tilt; adapt(measures, dt)
applies an exponential moving average with time constant BASELINE_TAU_S; it adapts
ONLY when no gesture is active AND near_neutral() holds (within BASELINE_GATE
of every threshold). Uncalibrated -> DEFAULT_BASELINE, no adaptation.
Wire it into Session with minimal edits. Tests: (a) small steady shift
(e.g. neck -8%) is absorbed after several simulated minutes with zero flags;
(b) a large sustained look-away is NOT absorbed and still flags; (c) a slow
creep that stays just under the gate boundary does not walk the baseline into
violation territory; (d) re-calibrate resets fully.
Report, STOP.
```

**STOP if:** test (b) or (c) fails. That is a cheating/false-negative hole and needs your decision on the gate. **CONTINUE trigger:** → `CONTINUE M6`.

---

### M6: Evidence Logger

**Goal:** Write one CSV row and one snapshot per flag. Nothing else. **Allowed files:** `invigilator/evidence.py`, `tests/test_evidence.py`. **Line budget:** \~70.

```text
MODULE M6: evidence.py.
class EvidenceLog(out_dir): creates the folder, appends to events.csv
(timestamp, gesture, label, held_seconds, snapshot), writes ONE JPEG per flag via
cv2.imwrite, flushes after each row, and has close().
Tests with tmp_path and a tiny numpy frame: header written once even across
re-opens; row count equals flag count; snapshot file exists; an unwritable folder
raises a clear error; calling record() never creates more than one image per call.
Add a privacy test that asserts no files other than events.csv and the .jpg
snapshots are ever created. Report, STOP.
```

**STOP if:** the code wants to buffer or store extra frames. **CONTINUE trigger:** → `CONTINUE M7`.

---

### M7: Camera Source

**Goal:** Camera auto-discovery behind a tiny interface so everything else can use a fake camera. **Allowed files:** `invigilator/camera.py`, `tests/test_camera.py`. **Line budget:** \~80.

```text
MODULE M7: camera.py.
Define a FrameSource protocol: read() -> (ok, frame), release(). Implement
open_camera(preferred, scan=CAMERA_SCAN): try preferred first, then others; an
index counts only if isOpened() AND a test read succeeds; raise a clear
SystemExit message pointing to README Troubleshooting if none work.
Also add FakeSource(frames) for tests and a --video file source for repeatable
offline replays (cv2.VideoCapture on a path).
Tests: use monkeypatching of cv2.VideoCapture to simulate index 0 failing and
index 2 working; all-fail case; FakeSource yields frames then ends.
Manual check (I run it): python -c "from invigilator.camera import open_camera; ...".
Report, STOP.
```

**STOP if:** the code attempts anything beyond local capture (network streams, RTSP). That is Phase 2 scope. **CONTINUE trigger:** the manual check shows my camera index is detected → `CONTINUE M8`.

---

### M8: Pose Estimator Wrapper

**Goal:** Isolate MediaPipe behind one class. **Allowed files:** `invigilator/pose.py`, `tests/test_pose.py`. **Line budget:** \~50.

```text
MODULE M8: pose.py.
class PoseEstimator(model_complexity=1): process(bgr_frame) -> (landmarks_list_or_None,
raw_result_for_drawing). Converts BGR->RGB, uses mp.solutions.pose.Pose, has close().
Fail early with a helpful message if mp.solutions is missing (wrong MediaPipe
version -> point to requirements.txt).
Tests: black frame -> (None, ...); close() is idempotent; a missing-solutions
error path via monkeypatch. Mark any test needing a person image as skipped.
Report, STOP.
```

**STOP if:** it tries to switch to the newer MediaPipe Tasks API (that needs a model download, which breaks the offline rule and needs your approval). **CONTINUE trigger:** → `CONTINUE M9`.

---

### M9: Overlay Renderer

**Goal:** Pure drawing function: gesture panel, timers, banners, FPS, calibration status. **Allowed files:** `invigilator/overlay.py`, `tests/test_overlay.py`. **Line budget:** \~80.

```text
MODULE M9: overlay.py.
render(frame, session_view, fps, now) -> frame. session_view is a small read-only
snapshot (active set, since{}, banners, calibrated flag) so overlay never mutates
Session. Colors: grey idle, amber active, red once past hold time.
Tests: output shape equals input shape; function does not mutate session_view;
smoke test with every gesture active does not raise.
Manual check: python tools/render_demo.py writes demo.png I can open.
Report, STOP.
```

**CONTINUE trigger:** I open `demo.png` and the panel looks readable → `CONTINUE M10`.

---

### M10: App Wiring and CLI

**Goal:** Assemble modules into the runnable app. No new logic. **Allowed files:** `invigilator/app.py`, `__main__.py`, `tests/test_app.py`. **Line budget:** \~90.

```text
MODULE M10: app.py.
main(): argparse (--camera, --out, --video, --no-snapshots), open_camera,
PoseEstimator, Session, EvidenceLog, overlay. Keys: c calibrate, s skeleton, q quit.
Use try/finally to release camera, pose and log. Mirror-flip the frame before pose.
Make the loop testable: run_loop(source, pose, session, evidence, render, keys=None)
so a test can feed FakeSource + a stub pose and assert flags get logged.
Tests: end-to-end with FakeSource and stub pose emitting "look_right" landmarks for
3 s of fake time -> exactly one CSV row and one snapshot.
Behavior must match gesture_poc.py. Report, STOP.
```

**Manual check:** `python -m invigilator` opens the camera; calibrate with `c`; raise a hand for 2 s and see a red flag banner and a new row in `evidence/events.csv`. **STOP if:** manual behavior differs from the legacy script in any visible way. **CONTINUE trigger:** manual check passes → `CONTINUE M11`.

---

### M11: Evaluation Harness (accuracy and false positives)

**Goal:** Measure against the PRD success criteria instead of guessing. **Allowed files:** `tools/eval_gestures.py`, `tools/soak_test.py`, `docs/EVAL.md`. **Line budget:** \~120.

```text
MODULE M11: evaluation tools.
1. tools/eval_gestures.py: guided script. For each of the 7 gestures it prompts me
   ("Raise your hand now"), counts 10 attempts, and records detected/not detected.
   Writes docs/EVAL.md-style table (gesture, hits/10).
2. tools/soak_test.py: runs for N minutes (default 2) while I sit and type
   normally; reports number of flags, FPS min/avg, and process memory at start/end.
Both reuse the app modules; no new detection logic. Output markdown. Report, STOP.
```

**PRD targets to check:** ≥ 8/10 per gesture; 0 flags in 2 min of normal sitting; ≥ 15 FPS; 30-min run without growth in memory. **STOP if:** any gesture is below 8/10 or the soak test shows a flag. Do NOT retune automatically. Propose changes and wait. **CONTINUE trigger:** targets met, or you've approved specific threshold changes and re-run → `CONTINUE M12`.

---

### M12: Docs and Release

**Goal:** Documentation matches reality; tag a release. **Allowed files:** `README.md`, `docs/`, `CHANGELOG.md`. **Line budget:** \~80.

```text
MODULE M12: docs and release.
Update README: setup, run, keys, gesture table (generated from config.py values),
troubleshooting (camera permissions per OS), test commands, project layout.
Add docs/EVAL.md results from M11, CHANGELOG.md, and a "Known limits" section
(single person, pose-based head direction, untuned thresholds, not proof of
misconduct). Run the full test suite and a fresh-venv install from
requirements.txt. Report, STOP.
```

**CONTINUE trigger:** fresh-venv install and `pytest` pass → tag `poc-v1.0`. The plan is complete.

---

## 4. Per-Module Safety Checklist (AI must confirm in every Checkpoint Report)

- [ ] Only allowed files were touched (`git diff --stat` pasted).
- [ ] All tests pass (`pytest -q` summary pasted).
- [ ] No new dependencies; no network code.
- [ ] No magic numbers outside `config.py`.
- [ ] No video written to disk beyond single flag snapshots.
- [ ] Pure modules contain no `time.time()`, `cv2`, or file I/O.
- [ ] Line budget respected (or explained).

## 5. Checkpoint Report Template

```text
CHECKPOINT: M<n> <name>
Built:      <files + line counts>
Tests:      <N passed / N failed>   (pasted summary)
Diff:       <git diff --stat>
Manual:     <exact command for me to run, expected result>
Risks:      <anything uncertain>
Waiting for: CONTINUE M<n+1> | STOP | REVERT M<n> | RESCOPE
```

## 6. Recovery Playbook

| Situation | Action |
| --- | --- |
| Tests fail twice | AI prints `HALT`, shows failing output; you decide: `RESCOPE`, or ask for a root-cause analysis only (no edits). |
| AI edits outside scope | `STOP`, then `REVERT Mx`; re-paste the module prompt and add "touch only: ". |
| AI drifts or gets verbose | `STATUS`, then re-paste the Master Rules Prompt. |
| Behavior differs from the legacy POC | Run the M0 characterization tests; fix the module that diverges before moving on. |
| You want a new feature | Do not slip it in. Finish M12, then open a new plan (candidates below). |

## 7. Out of Scope Until Approved (post-M12 backlog)

1. Phone/smartwatch detection (object detector, ONNX/INT8).
2. Face-landmark gaze estimation for better head direction.
3. Multiple people and per-seat zones.
4. Audio cues (whispering).
5. Port to dedicated edge hardware (Jetson-class) with quantized models.
6. Decentralized multi-node alerting (the v0.1 PRD), only if the product direction changes.

## 8. Definition of Done (whole project)

- All 13 modules complete, each with passing tests and an approved checkpoint.
- PRD criteria met: 7 gestures at ≥ 8/10, zero flags in the 2-minute normal-use soak, ≥ 15 FPS, 30-minute run stable.
- Fresh-environment install works from the README alone in under 10 minutes.
- Privacy verified: no network calls and only `events.csv` plus flag snapshots written.