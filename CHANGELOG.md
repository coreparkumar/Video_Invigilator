# Changelog

All notable changes to this project will be documented in this format.

## [1.0.0] - 2026-10-05
### Added
- **M0:** Baseline freeze and scaffold — frozen legacy `gesture_poc.py` as `legacy/gesture_poc.py`, created package structure, characterization tests (12 tests capturing legacy behavior)
- **M1:** Configuration module (`invigilator/config.py`) — all thresholds, gestures, constants in frozen dataclasses with validation (10 tests)
- **M2:** Feature extraction (`invigilator/features.py`) — scale-invariant measurements (yaw, neck, tilt) from landmarks, `near_neutral` gate (10 tests)
- **M3:** Gesture classifier (`invigilator/classifier.py`) — per-frame classification of 7 gestures + out_of_frame, priority rules, calibrated baseline shifts (21 tests)
- **M4:** Temporal session logic (`invigilator/session.py`) — hold timers, cooldowns, absence detection, flag events with injected time (16 tests)
- **M5:** Calibration & adaptive baseline (`invigilator/baseline.py`) — user-triggered calibration, EMA adaptation with safety gate (14 tests)
- **M6:** Evidence logger (`invigilator/evidence.py`) — CSV + one snapshot per flag, no extra files, context manager (9 tests + 1 skipped on Windows)
- **M7:** Camera source (`invigilator/camera.py`) — FrameSource protocol, auto-discovery, video file, FakeSource for testing (13 tests)
- **M8:** Pose estimator wrapper (`invigilator/pose.py`) — MediaPipe 0.10.14 solutions API wrapper, helpful error on wrong version (9 tests)
- **M9:** Overlay renderer (`invigilator/overlay.py`) — gesture panel, timers, banners, FPS, skeleton drawing, read-only session view (8 tests)
- **M10:** App wiring (`invigilator/app.py`, `__main__.py`) — CLI, main loop, integrates all modules, testable `run_loop` (3 tests)
- **M11:** Evaluation harness (`tools/eval_gestures.py`, `tools/soak_test.py`) — guided accuracy test (10 attempts/gesture), false-positive soak test
- **M12:** Documentation & release — updated README, EVAL.md template, CHANGELOG.md, .gitignore

### Changed
- Refactored from monolithic `gesture_poc.py` (311 lines) to 10 focused modules + tests
- All magic numbers moved to `config.py`
- Pure logic modules have no I/O, no `time.time()`, time injected as `now`
- Legacy behavior preserved and verified by characterization tests

### Fixed
- Bat file paths corrected (removed `scripts\` prefix)
- MediaPipe version pinned to 0.10.14 (solutions API)
- Python 3.12 venv for MediaPipe compatibility

### Tests
- **Total:** 125 passing, 1 skipped (Windows chmod)
- Characterization tests verify legacy behavior unchanged
- Each module has threshold "just under / just over" test pairs
- End-to-end test with FakeSource + stub pose

## [0.2.0] - 2026-10-05 (Legacy Baseline)
### Added
- Monolithic `gesture_poc.py` with all 7 gestures, calibration, temporal filtering, adaptive baseline, evidence logging, overlay
- Requirements: MediaPipe 0.10.14, OpenCV, NumPy<2

## [0.1.0] - 2026-10-05 (Initial Concept)
- Basic pose detection with MediaPipe
- Simple gesture rules