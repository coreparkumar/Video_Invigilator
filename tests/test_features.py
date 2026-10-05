"""Tests for features.py - pure feature extraction from landmarks."""
import math
from types import SimpleNamespace

from invigilator.features import measures, dist, near_neutral
from tests.conftest import make_landmarks


def test_neutral_pose_yaw_and_tilt_near_zero():
    """Neutral seated pose should have yaw ~0 and tilt ~0."""
    lm = make_landmarks()
    m = measures(lm)
    assert abs(m["yaw"]) < 0.1, f"Expected yaw ~0, got {m['yaw']}"
    assert abs(m["tilt"]) < 1.0, f"Expected tilt ~0, got {m['tilt']}"


def test_scale_invariance():
    """Doubling all coordinates should not change yaw, neck, tilt."""
    lm1 = make_landmarks()
    m1 = measures(lm1)

    # Double all coordinates from origin (0,0)
    lm2 = make_landmarks()
    for p in lm2:
        p.x *= 2
        p.y *= 2
    m2 = measures(lm2)

    assert abs(m1["yaw"] - m2["yaw"]) < 1e-6, "yaw not scale invariant"
    assert abs(m1["neck"] - m2["neck"]) < 1e-6, "neck not scale invariant"
    assert abs(m1["tilt"] - m2["tilt"]) < 1e-6, "tilt not scale invariant"


def test_tilt_abs_on_dx():
    """Tilt should use abs(dx) so swapping left/right doesn't change sign."""
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.65, y=0.45, visibility=1.0),  # L_SH on right
        12: SimpleNamespace(x=0.35, y=0.55, visibility=1.0),  # R_SH on left
    })
    m = measures(lm)
    # Right shoulder lower -> positive tilt
    assert m["tilt"] > 0, f"Expected positive tilt, got {m['tilt']}"


def test_zero_shoulder_width_no_divide_by_zero():
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


def test_dist_function():
    """dist() should compute Euclidean distance correctly."""
    a = SimpleNamespace(x=0.0, y=0.0, visibility=1.0)
    b = SimpleNamespace(x=3.0, y=4.0, visibility=1.0)
    assert dist(a, b) == 5.0


def test_near_neutral_true_when_well_inside():
    """near_neutral should return True when posture is well inside all thresholds."""
    m = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    assert near_neutral(m, base) is True


def test_near_neutral_false_when_yaw_exceeds_gate():
    """near_neutral should return False when yaw exceeds gate fraction."""
    m = {"yaw": 0.5, "neck": 0.65, "tilt": 0.0}  # yaw_shift=0.45, gate=0.5 -> limit=0.225
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    assert near_neutral(m, base) is False


def test_near_neutral_false_when_neck_below_gate():
    """near_neutral should return False when neck ratio below gate fraction."""
    # neck_ratio=0.55, gate=0.5 -> limit = 0.65 * (1 - 0.5 * 0.45) = 0.65 * 0.775 = 0.50375
    m = {"yaw": 0.0, "neck": 0.45, "tilt": 0.0}
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    assert near_neutral(m, base) is False


def test_near_neutral_false_when_tilt_exceeds_gate():
    """near_neutral should return False when tilt exceeds gate fraction."""
    # tilt_deg=14, gate=0.5 -> limit=7 deg
    m = {"yaw": 0.0, "neck": 0.65, "tilt": 10.0}
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    assert near_neutral(m, base) is False


def test_near_neutral_true_at_boundary():
    """near_neutral should return True at exact gate boundary (strict <)."""
    m = {"yaw": 0.224, "neck": 0.504, "tilt": 6.9}  # just under limits
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    assert near_neutral(m, base) is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])