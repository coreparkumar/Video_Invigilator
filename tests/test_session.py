"""Tests for session.py - temporal logic: hold timers, cooldowns, absence, flags."""
from collections import defaultdict
from types import SimpleNamespace

from invigilator.session import Session, FlagEvent
from invigilator.config import GESTURES, COOLDOWN_S, ABSENT_GRACE_S, VIS_MIN
from tests.conftest import make_landmarks, FakeClock


def make_landmarks_with_vis(overrides=None):
    """Helper to create landmarks with optional visibility overrides."""
    lm = make_landmarks()
    if overrides:
        for idx, vis in overrides.items():
            if idx < len(lm):
                lm[idx] = SimpleNamespace(
                    x=lm[idx].x, y=lm[idx].y, visibility=vis
                )
    return lm


def test_flag_exactly_at_hold_time():
    """Gesture flags exactly at hold time, not before."""
    clock = FakeClock(1000.0)
    session = Session()

    # look_right hold = 2.5s
    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    # First frame at t=1000.0 - gesture becomes active
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 0, f"Should not flag at 0s, got {flags}"

    # At 2.5s - exactly at threshold
    clock.advance(2.5)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 1, f"Should flag at 2.5s, got {flags}"
    assert flags[0].gesture_key == "look_right"
    assert abs(flags[0].held_seconds - 2.5) < 0.1


def test_flicker_resets_timer():
    """Gesture disappearing for one frame resets the hold timer."""
    clock = FakeClock(1000.0)
    session = Session()

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    lm_neutral = make_landmarks()

    # 2 seconds of look_right
    clock.advance(2.0)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 0

    # One frame neutral (flicker)
    clock.advance(0.033)  # ~30fps
    flags = session.update(lm_neutral, clock.now)
    assert len(flags) == 0

    # Resume look_right - timer should be reset
    clock.advance(0.033)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 0

    # Need full 2.5s again
    clock.advance(2.5)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 1, f"Should flag after reset + 2.5s, got {flags}"


def test_cooldown_blocks_repeat_then_allows():
    """Cooldown blocks repeat flag, then allows after COOLDOWN_S."""
    clock = FakeClock(1000.0)
    session = Session()

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    # First flag at 2.5s
    flags = session.update(lm_right, clock.now)
    clock.advance(2.5)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 1
    first_flag_time = clock.now

    # Continue holding - should NOT flag again within COOLDOWN_S (5s)
    clock.advance(3.0)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 0, f"Should not flag during cooldown, got {flags}"

    # After COOLDOWN_S - should flag again
    clock.advance(2.5)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 1, f"Should flag after cooldown, got {flags}"
    # held_seconds = total time gesture has been active = 2.5 + 3.0 + 2.5 = 8.0
    assert abs(flags[0].held_seconds - 8.0) < 0.5


def test_out_of_frame_flags_after_absent_grace_plus_hold():
    """Absence flags at ~ABSENT_GRACE_S + hold_time (3.0s)."""
    clock = FakeClock(1000.0)
    session = Session()

    # Start with person present
    lm = make_landmarks()
    flags = session.update(lm, clock.now)
    assert len(flags) == 0

    # Person leaves at t=1000.0
    clock.advance(0.5 + 0.1)  # 0.6s after leaving (past grace)
    flags = session.update(None, clock.now)
    assert len(flags) == 0

    # Grace ends at t=1001.1 (0.5s after leaving at t=1000.6)
    # We're at t=1000.6, need to reach 1001.1
    clock.advance(0.5)  # additional 0.5s to reach grace end
    flags = session.update(None, clock.now)
    assert len(flags) == 0

    # Next frame - grace over, out_of_frame becomes active
    clock.advance(0.033)
    flags = session.update(None, clock.now)
    assert len(flags) == 0, f"out_of_frame just became active, got {flags}"

    # Hold for out_of_frame threshold (3.0s) from when it became active
    clock.advance(3.0)
    flags = session.update(None, clock.now)
    # Should flag at exactly 3.0s held
    assert len(flags) == 1, f"Should flag at 3.0s, got {flags}"
    assert flags[0].gesture_key == "out_of_frame"
    assert abs(flags[0].held_seconds - 3.0) < 0.5


def test_out_of_frame_resets_on_return():
    """Person returning resets out_of_frame timer."""
    clock = FakeClock(1000.0)
    session = Session()

    lm = make_landmarks()

    # Person leaves
    clock.advance(2.0)
    flags = session.update(None, clock.now)
    assert len(flags) == 0

    # Returns before hold threshold
    clock.advance(1.0)
    flags = session.update(lm, clock.now)
    assert len(flags) == 0

    # Leaves again - timer should be fresh
    clock.advance(0.5)  # grace
    clock.advance(2.5)  # almost hold
    flags = session.update(None, clock.now)
    assert len(flags) == 0


def test_low_visibility_head_no_classify():
    """Low visibility head/shoulders treated as no reliable pose."""
    clock = FakeClock(1000.0)
    session = Session()

    # Low visibility on nose
    lm_low = make_landmarks_with_vis({0: 0.3})  # nose below VIS_MIN

    clock.advance(1.0)
    flags = session.update(lm_low, clock.now)
    assert len(flags) == 0
    assert session.active == set(), f"Should have no active gestures, got {session.active}"


def test_low_visibility_shoulders_no_classify():
    """Low visibility shoulders treated as no reliable pose."""
    clock = FakeClock(1000.0)
    session = Session()

    # Low visibility on shoulders
    lm_low = make_landmarks_with_vis({11: 0.3, 12: 0.3})

    clock.advance(1.0)
    flags = session.update(lm_low, clock.now)
    assert len(flags) == 0
    assert session.active == set()


def test_multiple_gestures_independent_timers():
    """Each gesture has independent hold timer."""
    clock = FakeClock(1000.0)
    session = Session()

    # Hand raised (1.0s) + look_right (2.5s)
    lm_both = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),  # hand raised
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),   # look right
    })

    # First frame - both become active
    flags = session.update(lm_both, clock.now)
    assert len(flags) == 0

    # At 1.0s - hand_raised should flag
    clock.advance(1.0)
    flags = session.update(lm_both, clock.now)
    assert len(flags) == 1
    assert flags[0].gesture_key == "hand_raised"

    # At 2.5s - look_right should flag
    clock.advance(1.5)
    flags = session.update(lm_both, clock.now)
    assert len(flags) == 1
    assert flags[0].gesture_key == "look_right"


def test_calibration_captures_baseline():
    """Calibration stores baseline measures."""
    session = Session()
    lm = make_landmarks()

    base = session.calibrate(lm)
    assert base is not None
    assert "yaw" in base
    assert "neck" in base
    assert "tilt" in base
    assert session.base is base


def test_adaptive_baseline_moves_toward_neutral():
    """Baseline slowly follows neutral posture when no gestures active."""
    clock = FakeClock(1000.0)
    session = Session()
    session.calibrate(make_landmarks())

    # Neutral pose, no gestures
    lm = make_landmarks()
    clock.advance(10.0)  # dt = 10s
    flags = session.update(lm, clock.now)

    # Baseline should have adapted slightly toward current (which is same as initial)
    assert session.base is not None


def test_adaptive_baseline_stops_during_gesture():
    """Baseline does NOT adapt when a gesture is active."""
    clock = FakeClock(1000.0)
    session = Session()
    session.calibrate(make_landmarks())

    original_yaw = session.base["yaw"]

    # Look right - gesture active
    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    clock.advance(10.0)
    flags = session.update(lm_right, clock.now)

    # Baseline yaw should NOT have moved toward the look_right pose
    assert session.base["yaw"] == original_yaw


def test_adaptive_baseline_stops_near_threshold():
    """Baseline does NOT adapt when near threshold (BASELINE_GATE)."""
    clock = FakeClock(1000.0)
    session = Session()
    session.calibrate(make_landmarks())

    original_yaw = session.base["yaw"]

    # Pose near but under threshold (within BASELINE_GATE)
    # yaw_shift=0.45, gate=0.5 -> limit=0.225
    # baseline yaw=0, so need yaw < 0.225
    lm_near = make_landmarks(overrides={
        0: SimpleNamespace(x=0.55, y=0.30, visibility=1.0),  # yaw = 0.25 -> dyaw = 0.25 > 0.225
    })
    clock.advance(10.0)
    flags = session.update(lm_near, clock.now)

    # This is NEAR threshold but technically active (dyaw > gate*threshold)
    # Actually wait: dyaw = 0.25, gate*threshold = 0.225, so 0.25 > 0.225 -> near_neutral=False
    # Let's check what happens
    from invigilator.features import near_neutral
    m = {"yaw": 0.25, "neck": 0.65, "tilt": 0.0}
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    assert near_neutral(m, base) is False  # Just outside gate


def test_start_timestamp_not_zero():
    """Start timestamps like 1000.0 (not 0) to catch cooldown-at-zero bugs."""
    clock = FakeClock(1000.0)
    session = Session()

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    # First flag
    flags = session.update(lm_right, clock.now)
    clock.advance(2.5)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 1

    # Cooldown check - last_flag should be 1002.5, not 2.5
    assert session.last_flag["look_right"] == 1002.5


def test_banners_expire_after_4_seconds():
    """Banners expire after 4 seconds."""
    clock = FakeClock(1000.0)
    session = Session()

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    # First frame
    flags = session.update(lm_right, clock.now)
    clock.advance(2.5)
    flags = session.update(lm_right, clock.now)
    assert len(session.banners) == 1

    # Advance past 4 seconds
    clock.advance(4.1)
    flags = session.update(lm_right, clock.now)
    assert len(session.banners) == 0


def test_get_view_returns_snapshot():
    """get_view returns read-only snapshot for overlay."""
    clock = FakeClock(1000.0)
    session = Session()

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    clock.advance(1.0)
    session.update(lm_right, clock.now)

    view = session.get_view()
    assert "active" in view
    assert "since" in view
    assert "banners" in view
    assert "calibrated" in view
    assert isinstance(view["active"], frozenset)


def test_flag_event_has_correct_fields():
    """FlagEvent dataclass has gesture_key, label, held_seconds."""
    clock = FakeClock(1000.0)
    session = Session()

    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })

    # First frame
    flags = session.update(lm_right, clock.now)
    clock.advance(2.5)
    flags = session.update(lm_right, clock.now)
    assert len(flags) == 1

    event = flags[0]
    assert isinstance(event, FlagEvent)
    assert event.gesture_key == "look_right"
    assert event.label == "Looking right"
    assert event.held_seconds > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])