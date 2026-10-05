"""Characterization tests: capture current behavior of the legacy POC."""
import sys
sys.path.insert(0, 'legacy')

from gesture_poc import classify, measures, GESTURES, THRESHOLDS, DEFAULT_BASELINE, VIS_MIN
from tests.conftest import make_landmarks, SimpleNamespace


def test_neutral_pose_returns_empty_set():
    """Neutral seated pose should have no active gestures."""
    lm = make_landmarks()
    active = classify(lm)
    assert active == set(), f"Expected empty set, got {active}"


def test_hand_raised_when_wrist_above_nose():
    """Wrist above nose by threshold should flag hand_raised."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),  # L_WR raised
    })
    active = classify(lm)
    assert "hand_raised" in active, f"Expected hand_raised, got {active}"


def test_look_right_when_nose_shifted_right():
    """Nose shifted right of ear midpoint should flag look_right."""
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),  # NOSE shifted right
    })
    active = classify(lm)
    assert "look_right" in active, f"Expected look_right, got {active}"


def test_look_left_when_nose_shifted_left():
    """Nose shifted left of ear midpoint should flag look_left."""
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.38, y=0.30, visibility=1.0),  # NOSE shifted left
    })
    active = classify(lm)
    assert "look_left" in active, f"Expected look_left, got {active}"


def test_leaning_when_shoulder_tilted():
    """Shoulder line tilted beyond threshold should flag leaning."""
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.45, visibility=1.0),  # L_SH up
        12: SimpleNamespace(x=0.65, y=0.55, visibility=1.0),  # R_SH down
    })
    active = classify(lm)
    assert "leaning" in active, f"Expected leaning, got {active}"


def test_look_down_when_neck_shortened():
    """Neck ratio below threshold should flag look_down."""
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.42, visibility=1.0),  # NOSE lowered
    })
    active = classify(lm)
    assert "look_down" in active, f"Expected look_down, got {active}"


def test_hand_to_face_when_wrist_near_nose():
    """Wrist close to nose should flag hand_to_face (priority over ear)."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.48, y=0.32, visibility=1.0),  # L_WR near nose
    })
    active = classify(lm)
    assert "hand_to_face" in active, f"Expected hand_to_face, got {active}"


def test_hand_to_ear_when_wrist_near_ear_not_above_nose():
    """Wrist near ear (not above nose) should flag hand_to_ear."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.38, y=0.30, visibility=1.0),  # L_WR near left ear
    })
    active = classify(lm)
    assert "hand_to_ear" in active, f"Expected hand_to_ear, got {active}"


def test_low_visibility_wrist_ignored():
    """Wrist below VIS_MIN should be ignored."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=0.3),  # L_WR low visibility
    })
    active = classify(lm)
    assert "hand_raised" not in active, f"Low-vis wrist should not trigger, got {active}"


def test_measures_scale_invariance():
    """Doubling all coordinates should not change yaw, neck, tilt (scale invariance)."""
    lm1 = make_landmarks()
    m1 = measures(lm1)

    # Double all coordinates from origin (0,0)
    lm2 = make_landmarks()
    for p in lm2:
        p.x *= 2
        p.y *= 2
    m2 = measures(lm2)

    assert abs(m1["yaw"] - m2["yaw"]) < 1e-6
    assert abs(m1["neck"] - m2["neck"]) < 1e-6
    assert abs(m1["tilt"] - m2["tilt"]) < 1e-6


def test_measures_tilt_abs_on_dx():
    """Tilt should use abs(dx) so swapping left/right doesn't change sign."""
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.65, y=0.45, visibility=1.0),  # L_SH on right
        12: SimpleNamespace(x=0.35, y=0.55, visibility=1.0),  # R_SH on left
    })
    m = measures(lm)
    # Should be positive tilt (right shoulder lower)
    assert m["tilt"] > 0


def test_measures_no_divide_by_zero():
    """Zero shoulder width should not crash (returns safe defaults)."""
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.5, y=0.5, visibility=1.0),
        12: SimpleNamespace(x=0.5, y=0.5, visibility=1.0),
    })
    m = measures(lm)
    assert m["sh_w"] > 0
    assert "yaw" in m
    assert "neck" in m
    assert "tilt" in m


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])