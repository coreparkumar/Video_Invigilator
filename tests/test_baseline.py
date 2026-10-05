"""Tests for baseline.py - calibration and adaptive baseline."""
import math
from types import SimpleNamespace

from invigilator.baseline import Baseline
from invigilator.config import BASELINE_TAU_S, BASELINE_GATE, DEFAULT_BASELINE
from invigilator.features import measures, near_neutral
from tests.conftest import make_landmarks, FakeClock


def test_uncalibrated_returns_default():
    """Before calibration, baseline returns DEFAULT_BASELINE."""
    baseline = Baseline()
    assert baseline.base is None
    assert baseline.is_calibrated is False
    assert baseline.get_default() == DEFAULT_BASELINE


def test_calibrate_stores_measures():
    """Calibration stores yaw, neck, tilt from current landmarks."""
    baseline = Baseline()
    lm = make_landmarks()
    base = baseline.calibrate(lm)

    assert base is not None
    assert baseline.is_calibrated is True
    assert "yaw" in base
    assert "neck" in base
    assert "tilt" in base
    assert "sh_w" not in base  # shoulder width not stored
    assert baseline.base is base


def test_recalibrate_resets_fully():
    """Re-calibrate resets baseline fully to new posture."""
    baseline = Baseline()
    lm1 = make_landmarks()
    baseline.calibrate(lm1)
    original_yaw = baseline.base["yaw"]

    # Different posture
    lm2 = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    base = baseline.calibrate(lm2)

    assert base["yaw"] != original_yaw
    assert base["yaw"] > 0.5  # shifted right


def test_reset_clears_calibration():
    """Reset returns to uncalibrated state."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())
    assert baseline.is_calibrated

    baseline.reset()
    assert not baseline.is_calibrated
    assert baseline.base is None


def test_adapt_moves_toward_current():
    """Adapt slowly moves baseline toward current posture."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())
    original_yaw = baseline.base["yaw"]

    # New posture with slightly different yaw
    m = {"yaw": original_yaw + 0.1, "neck": 0.65, "tilt": 0.0}
    baseline.adapt(m, dt=BASELINE_TAU_S)  # dt = tau -> alpha = 1 - 1/e ≈ 0.63

    # Should have moved ~63% of the way
    expected = original_yaw + 0.1 * (1.0 - math.exp(-1.0))
    assert abs(baseline.base["yaw"] - expected) < 0.001


def test_adapt_only_when_calibrated():
    """Adapt does nothing when uncalibrated."""
    baseline = Baseline()
    m = {"yaw": 0.5, "neck": 0.65, "tilt": 0.0}
    baseline.adapt(m, dt=10.0)  # Should not crash
    assert baseline.base is None


def test_can_adapt_true_when_near_neutral():
    """can_adapt returns True when calibrated and near neutral."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())

    m = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}  # exactly at baseline
    assert baseline.can_adapt(m) is True


def test_can_adapt_false_when_not_calibrated():
    """can_adapt returns False when uncalibrated."""
    baseline = Baseline()
    m = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
    assert baseline.can_adapt(m) is False


def test_can_adapt_false_when_yaw_exceeds_gate():
    """can_adapt returns False when yaw exceeds gate fraction."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())

    # yaw_shift=0.45, gate=0.5 -> limit=0.225
    m = {"yaw": 0.3, "neck": 0.65, "tilt": 0.0}
    assert baseline.can_adapt(m) is False


def test_can_adapt_false_when_neck_below_gate():
    """can_adapt returns False when neck ratio below gate."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())

    # neck_ratio=0.55, gate=0.5 -> limit = 0.65 * (1 - 0.5*0.45) = 0.50375
    m = {"yaw": 0.0, "neck": 0.45, "tilt": 0.0}
    assert baseline.can_adapt(m) is False


def test_can_adapt_false_when_tilt_exceeds_gate():
    """can_adapt returns False when tilt exceeds gate fraction."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())

    # tilt_deg=14, gate=0.5 -> limit=7
    m = {"yaw": 0.0, "neck": 0.65, "tilt": 10.0}
    assert baseline.can_adapt(m) is False


def test_small_steady_shift_absorbed_after_minutes():
    """Small steady shift (e.g. neck -8%) absorbed after simulated minutes with zero flags."""
    clock = FakeClock(1000.0)
    baseline = Baseline()
    baseline.calibrate(make_landmarks())
    original_neck = baseline.base["neck"]

    # Simulate slow drift: neck decreases 8% over several minutes
    # At each step, posture is near neutral (no gestures active)
    for i in range(60):  # 60 steps
        clock.advance(10.0)  # 10 seconds per step
        drift = i * 0.001  # small drift per step
        lm = make_landmarks(overrides={
            0: SimpleNamespace(x=0.50, y=0.30 + drift, visibility=1.0),  # neck decreases
        })
        m = measures(lm)

        if baseline.can_adapt(m):
            baseline.adapt(m, 10.0)

    # After many steps, baseline should have absorbed most of the drift
    assert baseline.base["neck"] < original_neck
    # Should be close to final posture (which had neck ~ original - 0.06)
    final_neck = measures(make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.30 + 0.06, visibility=1.0),
    }))["neck"]
    assert abs(baseline.base["neck"] - final_neck) < 0.05


def test_large_sustained_lookaway_not_absorbed():
    """Large sustained look-away is NOT absorbed and still flags."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())

    # Large look-away: yaw = 0.8 (well beyond threshold 0.45)
    m = {"yaw": 0.8, "neck": 0.65, "tilt": 0.0}

    # Not near neutral -> can_adapt = False
    assert baseline.can_adapt(m) is False

    # Baseline should not change
    original_yaw = baseline.base["yaw"]
    baseline.adapt(m, dt=100.0)  # Even with large dt
    assert baseline.base["yaw"] == original_yaw


def test_slow_creep_under_gate_does_not_walk_into_violation():
    """Slow creep staying just under gate boundary does not walk baseline into violation."""
    baseline = Baseline()
    baseline.calibrate(make_landmarks())

    # Creep yaw slowly, staying just inside gate boundary
    # gate * yaw_shift = 0.5 * 0.45 = 0.225
    for i in range(20):
        yaw = 0.22  # Just under gate
        m = {"yaw": yaw, "neck": 0.65, "tilt": 0.0}
        assert baseline.can_adapt(m) is True  # Still near neutral
        baseline.adapt(m, dt=10.0)

    # Baseline should have moved toward 0.22 but not past it
    assert baseline.base["yaw"] < 0.225  # Still within threshold boundary
    # But should NOT have moved all the way to violation territory (>0.45)
    assert baseline.base["yaw"] < 0.45


if __name__ == "__main__":
    import math
    import pytest
    pytest.main([__file__, "-v"])