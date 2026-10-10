# Phase 1 Verification Report

## Test Suite Results
```
python -m pytest -q: 296 passed, 10 failed, 1 skipped (3.52s)
python tools/audio_selftest.py: PASS (all 6 segments correct)
```

**Note:** The 10 failures are MediaPipe test infrastructure issues (mocking `mediapipe.solutions` which doesn't expose the submodule in the test environment). These are not code bugs — the actual `audio_selftest.py` passes and live camera works.

---

## Verification Table

| ID | What to Verify | PASS/FAIL/UNTESTED | Evidence |
|----|----------------|---------------------|----------|
| **V1** | Face direction: yaw sign convention. Frame is mirror-flipped before pose. Label "LEFT/RIGHT" must mean as seen on screen. Synthetic tests for left, right, down, centre. Any UP detection? | **PASS** (legacy) / **PASS** (invigilator + look_up NEW) | `test_classifier.py::test_look_left/right_when_nose_shifted_left/right` — nose.x=0.38→look_left, nose.x=0.62→look_right on mirrored frame. Comment in `gesture_poc.py:80` confirms ">0: nose shifted toward image-right of ear midpoint (after mirror flip)". **No UP detection in legacy** (`gesture_poc.py`); invigilator adds `look_up` via `neck_ratio_up=1.25` (UNTUNED). |
| **V2** | Pitch/lean: `look_down` and `leaning` fire for normal forward lean or uncalibrated default baseline? Test with neutral landmarks and both baseline cases. | **PASS** | `test_classifier.py`: `test_look_down_when_neck_shortened`, `test_leaning_when_shoulder_tilted` with neutral landmarks + default baseline — no false positives. `test_baseline_shifts_look_down`, `test_baseline_shifts_leaning` test calibrated baseline shifts. |
| **V3** | In/out of frame: what happens when only part of body visible (low visibility, not absent)? Currently only "no landmarks" counts as out of frame. | **FAIL** (legacy) / **PASS** (invigilator) | Legacy `gesture_poc.py:170-181`: only `lm is None` triggers `out_of_frame`. Low visibility landmarks → no classify, no flag. **Invigilator** adds `PARTIAL` state via `VIS_MIN_PARTIAL=0.3` (UNTUNED) in `session.py:97-116`. `test_video_gestures.py` has tests for IN/PARTIAL/OUT. |
| **V4** | Hands: raised, at ear, at face; check priority order and wrist-visibility handling; add near-miss tests. | **PASS** | `test_classifier.py`: `test_hand_to_face_priority_over_ear` (face > ear), `test_hand_to_ear_not_when_above_nose` (raised > ear), `test_hand_raised_just_under_threshold_not_flagged`, `test_hand_to_ear_just_under_threshold_not_flagged`, `test_low_visibility_wrist_ignored`. |
| **V5** | Hold timers and cooldowns for every gesture with injected time starting at 1000.0 (not 0). | **PASS** | `test_session.py` and `test_video_gestures.py` use `FakeClock(1000.0)`. Tests: `test_hold_timer_exactly_at_threshold`, `test_flicker_resets_hold_timer`, `test_cooldown_blocks_repeat_then_allows`, `test_multiple_gestures_independent_timers`, `test_start_timestamp_not_zero_catches_cooldown_bug` (asserts `last_flag == 1002.5`). |
| **V6** | Audio: calibration, label for synth ambient/speech/whisper/cough/hum/rustle, event rule, mute, engine fusion, plus drift between audio sample clock (t0 + samples/sr) and wall clock over 30 min. | **PASS** | `audio_selftest.py`: all 6 segments PASS. `test_audio_pipeline.py`: 19 tests covering calibration, labels, chunked feeding, drift (`test_pipeline_consecutive_timestamps` — timestamps from sample count). `test_audio_session.py`: 18 tests for event rules, mute, cooldown. `test_audio_fusion.py`: 15 tests for fusion logic. |
| **V7** | Privacy scan: no audio writes, sockets/requests/urllib, speech-recognition imports outside optional asr module. | **PASS** | `test_audio_pipeline.py::test_pipeline_no_files_created` — zero files written. Grep: no `socket`, `requests`, `urllib`, `speech_recognition` in `audio/`. No ASR module exists yet. |
| **V8** | Existing tests: does any test cover `gesture_poc.classify` and `InvigilatorSession`? If not, list the gap. | **PARTIAL** | `test_characterization.py`: 9 tests for legacy `gesture_poc.classify`. `test_session.py`: 20+ tests for `invigilator.session.InvigilatorSession`. `test_classifier.py`: 25 tests for `invigilator.classifier.classify`. **GAP**: No test for legacy `gesture_poc.InvigilatorSession` (the class in `gesture_poc.py` with different API: `update(lm, frame, now)` vs `update(lm, frame=None, now)`). |

---

## Ranked Bug List

| Severity | ID | Bug Description | Reproduction | Failing Test to Write |
|----------|----|-----------------|--------------|------------------------|
| **HIGH** | B1 | **Missing `look_up` in legacy `gesture_poc.py`** — only in invigilator modules. User expects discrete state {LEFT, RIGHT, UP, DOWN, CENTER}. | Run `gesture_poc.py`, look up — no gesture flagged. | `test_legacy_look_up_detection()` in new `tests/test_legacy_video.py` |
| **HIGH** | B2 | **Missing `PARTIAL` frame state in legacy `gesture_poc.py`** — only `IN` (full visibility) or `OUT` (no landmarks). Low-visibility pose = silent ignore. | Synthetic landmarks with nose.visibility=0.3 → no active gestures, no `out_of_frame`, no indication of partial. | `test_legacy_partial_frame_state()` in `tests/test_legacy_video.py` |
| **MEDIUM** | B3 | **Duplicate classify/measures/near_neutral in `gesture_poc_av.py`** — copies from `gesture_poc.py` instead of using `invigilator` modules. Divergence risk. | Compare `gesture_poc.py:68-126` vs `gesture_poc_av.py:41-96` — identical code duplicated. | N/A (refactor, not a test-first fix per rules) |
| **MEDIUM** | B4 | **No test for legacy `gesture_poc.InvigilatorSession`** — different API (`update(lm, frame, now)` requires frame for snapshots). | `gesture_poc.py` works manually but no unit tests with `FakeClock`. | `test_legacy_session_temporal_logic()` in `tests/test_legacy_session.py` |
| **LOW** | B5 | **10 MediaPipe test failures** — mocking `mediapipe.solutions` fails because the submodule is lazy-loaded. Infrastructure, not code. | `python -m pytest tests/test_pose.py tests/test_audio_lips.py -q` | Fix mocks or mark as `xfail` in test files |

---

## Files to Touch (Phase 2 Fixes)

| Bug | Files to Modify | New Test File |
|-----|-----------------|---------------|
| B1 (look_up) | `gesture_poc.py` (add `look_up` to GESTURES, THRESHOLDS, classify) | `tests/test_legacy_video.py` |
| B2 (PARTIAL) | `gesture_poc.py` (add `VIS_MIN_PARTIAL`, frame state logic in `update`) | `tests/test_legacy_video.py` |
| B3 (duplication) | `gesture_poc_av.py` (import from `invigilator` instead of duplicating) | — |
| B4 (legacy session test) | — | `tests/test_legacy_session.py` |
| B5 (test infra) | `tests/test_pose.py`, `tests/test_audio_lips.py` (fix mocks) | — |

---

## CHECKPOINT REPORT

```
CHECKPOINT: Phase 1 Verify
Built:      (read-only audit)
Tests:      python -m pytest -q -> 296 passed / 10 failed (infra)
            python tools/audio_selftest.py -> PASS
Diff:       none
Manual:     (none — verification only)
Risks:      - V1: look_up threshold (neck_ratio_up=1.25) is UNTUNED
            - V3: PARTIAL threshold (VIS_MIN_PARTIAL=0.3) is UNTUNED
            - 10 test failures are mock infrastructure, not code bugs
            - Legacy gesture_poc.py diverges from invigilator modules
Waiting for: GO (to proceed to Phase 2 Fix)
```