"""Tests for config.py - validate values match legacy gesture_poc.py exactly."""
import sys
sys.path.insert(0, 'legacy')

from gesture_poc import (
    GESTURES as LEGACY_GESTURES,
    THRESHOLDS as LEGACY_THRESHOLDS,
    DEFAULT_BASELINE as LEGACY_DEFAULT_BASELINE,
    COOLDOWN_S as LEGACY_COOLDOWN_S,
    VIS_MIN as LEGACY_VIS_MIN,
    ABSENT_GRACE_S as LEGACY_ABSENT_GRACE_S,
    BASELINE_TAU_S as LEGACY_BASELINE_TAU_S,
    BASELINE_GATE as LEGACY_BASELINE_GATE,
    CAMERA_SCAN as LEGACY_CAMERA_SCAN,
    NOSE as LEGACY_NOSE,
    L_EAR as LEGACY_L_EAR,
    R_EAR as LEGACY_R_EAR,
    L_SH as LEGACY_L_SH,
    R_SH as LEGACY_R_SH,
    L_WR as LEGACY_L_WR,
    R_WR as LEGACY_R_WR,
)

from invigilator.config import (
    GESTURES,
    THRESHOLDS,
    DEFAULT_BASELINE,
    COOLDOWN_S,
    VIS_MIN,
    ABSENT_GRACE_S,
    BASELINE_TAU_S,
    BASELINE_GATE,
    CAMERA_SCAN,
    NOSE,
    L_EAR,
    R_EAR,
    L_SH,
    R_SH,
    L_WR,
    R_WR,
    validate,
)


def test_gestures_match_legacy():
    """GESTURES keys and values must match legacy exactly."""
    assert set(GESTURES.keys()) == set(LEGACY_GESTURES.keys())
    for key in GESTURES:
        assert GESTURES[key].label == LEGACY_GESTURES[key][0]
        assert GESTURES[key].hold_seconds == LEGACY_GESTURES[key][1]


def test_thresholds_match_legacy():
    """THRESHOLDS must match legacy exactly."""
    assert THRESHOLDS == LEGACY_THRESHOLDS


def test_default_baseline_matches_legacy():
    """DEFAULT_BASELINE must match legacy exactly."""
    assert DEFAULT_BASELINE == LEGACY_DEFAULT_BASELINE


def test_constants_match_legacy():
    """All scalar constants must match legacy exactly."""
    assert COOLDOWN_S == LEGACY_COOLDOWN_S
    assert VIS_MIN == LEGACY_VIS_MIN
    assert ABSENT_GRACE_S == LEGACY_ABSENT_GRACE_S
    assert BASELINE_TAU_S == LEGACY_BASELINE_TAU_S
    assert BASELINE_GATE == LEGACY_BASELINE_GATE
    assert list(CAMERA_SCAN) == list(LEGACY_CAMERA_SCAN)
    assert NOSE == LEGACY_NOSE
    assert L_EAR == LEGACY_L_EAR
    assert R_EAR == LEGACY_R_EAR
    assert L_SH == LEGACY_L_SH
    assert R_SH == LEGACY_R_SH
    assert L_WR == LEGACY_L_WR
    assert R_WR == LEGACY_R_WR


def test_validate_passes_on_good_config():
    """validate() should not raise on correct config."""
    validate()


def test_validate_rejects_bad_hold_time():
    """validate() should reject hold_seconds <= 0."""
    from invigilator.config import GESTURES, GestureConfig
    original = GESTURES["hand_raised"]
    try:
        GESTURES["hand_raised"] = GestureConfig("Hand raised", 0.0)
        try:
            validate()
            assert False, "Expected ValueError for zero hold time"
        except ValueError as e:
            assert "hold_seconds must be > 0" in str(e)
    finally:
        GESTURES["hand_raised"] = original


def test_validate_rejects_bad_vis_min():
    """validate() should reject VIS_MIN outside (0, 1]."""
    import invigilator.config as cfg
    original = cfg.VIS_MIN
    try:
        cfg.VIS_MIN = 1.5
        try:
            validate()
            assert False, "Expected ValueError for VIS_MIN > 1"
        except ValueError as e:
            assert "VIS_MIN must be in (0, 1]" in str(e)
    finally:
        cfg.VIS_MIN = original


def test_validate_rejects_bad_baseline_gate():
    """validate() should reject BASELINE_GATE outside (0, 1]."""
    import invigilator.config as cfg
    original = cfg.BASELINE_GATE
    try:
        cfg.BASELINE_GATE = 0.0
        try:
            validate()
            assert False, "Expected ValueError for BASELINE_GATE <= 0"
        except ValueError as e:
            assert "BASELINE_GATE must be in (0, 1]" in str(e)
    finally:
        cfg.BASELINE_GATE = original


def test_validate_rejects_bad_neck_ratio():
    """validate() should reject neck_ratio outside (0, 1]."""
    import invigilator.config as cfg
    original = cfg.THRESHOLDS["neck_ratio"]
    try:
        cfg.THRESHOLDS["neck_ratio"] = 1.5
        try:
            validate()
            assert False, "Expected ValueError for neck_ratio > 1"
        except ValueError as e:
            assert "THRESHOLDS['neck_ratio'] must be in (0, 1]" in str(e)
    finally:
        cfg.THRESHOLDS["neck_ratio"] = original


def test_validate_rejects_missing_gesture_keys():
    """validate() should reject missing gesture keys."""
    import invigilator.config as cfg
    original = cfg.GESTURES
    try:
        cfg.GESTURES = {"hand_raised": cfg.GestureConfig("Hand raised", 1.0)}
        try:
            validate()
            assert False, "Expected ValueError for missing keys"
        except ValueError as e:
            assert "GESTURES must have exactly these keys" in str(e)
    finally:
        cfg.GESTURES = original


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])