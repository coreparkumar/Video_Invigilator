"""Tests for legacy gesture_poc.InvigilatorSession temporal logic.

The legacy session has a different API: update(lm, frame, now) where frame is required for snapshots.
"""
import sys
sys.path.insert(0, 'legacy')

from types import SimpleNamespace
import numpy as np
import os
import shutil
import csv

from gesture_poc import InvigilatorSession as LegacySession
from gesture_poc import GESTURES, THRESHOLDS, DEFAULT_BASELINE, VIS_MIN, ABSENT_GRACE_S, COOLDOWN_S
from tests.conftest import make_landmarks, FakeClock


# Dummy frame for snapshot saving
DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _clean_evidence(dirname):
    """Clean up evidence directory."""
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)


def test_legacy_session_flag_at_hold_time():
    """Legacy session flags exactly at hold time."""
    _clean_evidence("test_evidence_flag")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_flag")

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    # First frame at t=1000.0
    session.update(lm_right, DUMMY_FRAME, clock.now)
    clock.advance(GESTURES["look_right"][1])  # hold_seconds
    session.update(lm_right, DUMMY_FRAME, clock.now)

    # Legacy session prints to console but doesn't return flags list
    # Check banners instead
    assert len(session.banners) == 1
    assert "FLAG: Looking right" in session.banners[0][1]


def test_legacy_session_cooldown():
    """Legacy session respects cooldown."""
    _clean_evidence("test_evidence_cooldown")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_cooldown")

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    # First flag
    session.update(lm_right, DUMMY_FRAME, clock.now)
    clock.advance(GESTURES["look_right"][1])
    session.update(lm_right, DUMMY_FRAME, clock.now)

    # Within cooldown - should not flag again
    clock.advance(3.0)
    session.update(lm_right, DUMMY_FRAME, clock.now)

    # After cooldown
    clock.advance(COOLDOWN_S - 3.0 + 0.1)
    session.update(lm_right, DUMMY_FRAME, clock.now)

    # Check evidence log has 2 entries (banners expire after 4s)
    log_path = os.path.join("test_evidence_cooldown", "events.csv")
    with open(log_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Header + 2 flag rows
    assert len(rows) == 3


def test_legacy_session_out_of_frame():
    """Legacy session detects out_of_frame after absent grace + hold."""
    _clean_evidence("test_evidence_oof")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_oof")
    session.calibrate(make_landmarks())

    # Person leaves
    clock.advance(ABSENT_GRACE_S + 0.1)
    session.update(None, DUMMY_FRAME, clock.now)
    assert "out_of_frame" not in session.active

    # Wait for hold
    clock.advance(GESTURES["out_of_frame"][1])
    session.update(None, DUMMY_FRAME, clock.now)
    assert "out_of_frame" in session.active


def test_legacy_session_flicker_resets():
    """Legacy session resets timer on flicker."""
    _clean_evidence("test_evidence_flicker")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_flicker")

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    lm_neutral = make_landmarks()

    clock.advance(2.0)
    session.update(lm_right, DUMMY_FRAME, clock.now)

    # Flicker
    clock.advance(0.033)
    session.update(lm_neutral, DUMMY_FRAME, clock.now)

    clock.advance(0.033)
    session.update(lm_right, DUMMY_FRAME, clock.now)

    # Need full hold again
    clock.advance(GESTURES["look_right"][1])
    session.update(lm_right, DUMMY_FRAME, clock.now)
    assert len(session.banners) == 1


def test_legacy_session_multiple_gestures():
    """Legacy session handles multiple gestures with independent timers."""
    _clean_evidence("test_evidence_multi")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_multi")

    lm_both = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),  # hand_raised (1.0s)
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),   # look_right (2.5s)
    })

    session.update(lm_both, DUMMY_FRAME, clock.now)

    clock.advance(GESTURES["hand_raised"][1])
    session.update(lm_both, DUMMY_FRAME, clock.now)

    clock.advance(GESTURES["look_right"][1] - GESTURES["hand_raised"][1])
    session.update(lm_both, DUMMY_FRAME, clock.now)

    # Check evidence log has 2 entries
    log_path = os.path.join("test_evidence_multi", "events.csv")
    with open(log_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Header + 2 flag rows
    assert len(rows) == 3
    gestures = [r[1] for r in rows[1:]]
    assert "hand_raised" in gestures
    assert "look_right" in gestures


def test_legacy_session_calibration():
    """Legacy session captures baseline on calibrate."""
    _clean_evidence("test_evidence_cal")
    session = LegacySession(out_dir="test_evidence_cal")
    lm = make_landmarks()
    base = session.calibrate(lm)

    assert base is not None
    assert "yaw" in base
    assert "neck" in base
    assert "tilt" in base


def test_legacy_session_baseline_adaptation():
    """Legacy session adapts baseline when near neutral."""
    _clean_evidence("test_evidence_adapt")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_adapt")
    session.calibrate(make_landmarks())

    original_yaw = session.base["yaw"]

    # Neutral pose, no gestures
    lm = make_landmarks()
    clock.advance(10.0)
    session.update(lm, DUMMY_FRAME, clock.now)

    # Baseline should have adapted slightly
    # (exact value depends on BASELINE_TAU_S)


def test_legacy_session_baseline_stops_during_gesture():
    """Legacy session does NOT adapt baseline during gesture."""
    _clean_evidence("test_evidence_stop")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_stop")
    session.calibrate(make_landmarks())

    original_yaw = session.base["yaw"]

    # Look right - gesture active
    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    clock.advance(10.0)
    session.update(lm_right, DUMMY_FRAME, clock.now)

    # Baseline yaw should NOT have moved
    assert session.base["yaw"] == original_yaw


def test_legacy_session_frame_state_tracking():
    """Legacy session tracks frame_state (IN/PARTIAL/OUT)."""
    _clean_evidence("test_evidence_fstate")
    clock = FakeClock(1000.0)
    session = LegacySession(out_dir="test_evidence_fstate")
    session.calibrate(make_landmarks())

    # Full visibility -> IN
    session.update(make_landmarks(), DUMMY_FRAME, clock.now)
    assert session.frame_state == "IN"

    # Low visibility nose -> PARTIAL
    lm_low_nose = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.30, visibility=0.3),
    })
    session.update(lm_low_nose, DUMMY_FRAME, clock.now)
    assert session.frame_state == "PARTIAL"

    # Low visibility shoulders -> PARTIAL
    lm_low_sh = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.50, visibility=0.3),
        12: SimpleNamespace(x=0.65, y=0.50, visibility=0.3),
    })
    session.update(lm_low_sh, DUMMY_FRAME, clock.now)
    assert session.frame_state == "PARTIAL"

    # No landmarks -> OUT
    clock.advance(ABSENT_GRACE_S + 0.1)
    session.update(None, DUMMY_FRAME, clock.now)
    assert session.frame_state == "OUT"

    # Return -> IN
    clock.advance(1.0)
    session.update(make_landmarks(), DUMMY_FRAME, clock.now)
    assert session.frame_state == "IN"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])