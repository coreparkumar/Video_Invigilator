"""Tests for classifier.py - pure gesture classification from landmarks + baseline."""
from types import SimpleNamespace

from invigilator.classifier import classify
from invigilator.config import GESTURES, THRESHOLDS, DEFAULT_BASELINE, VIS_MIN
from tests.conftest import make_landmarks


def test_neutral_pose_returns_empty_set():
    """Neutral seated pose should have no active gestures."""
    lm = make_landmarks()
    active = classify(lm)
    assert active == set(), f"Expected empty set, got {active}"


def test_hand_raised_when_wrist_above_nose():
    """Wrist above nose by threshold should flag hand_raised."""
    # wrist y < nose.y - hand_raised_above_nose * shoulder_width
    # shoulder_width ~ 0.3, threshold = 0.15 -> need wrist ~0.045 above nose
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),  # L_WR raised
    })
    active = classify(lm)
    assert "hand_raised" in active, f"Expected hand_raised, got {active}"


def test_hand_raised_just_under_threshold_not_flagged():
    """Wrist just under threshold should NOT flag hand_raised."""
    # nose.y = 0.30, sh_w = 0.3, threshold = 0.15 * 0.3 = 0.045
    # need wr.y >= 0.30 - 0.045 = 0.255 to NOT trigger
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.26, visibility=1.0),  # just under
    })
    active = classify(lm)
    assert "hand_raised" not in active, f"Expected no hand_raised, got {active}"


def test_hand_to_face_when_wrist_near_nose():
    """Wrist close to nose should flag hand_to_face (priority over ear)."""
    # dist(wrist, nose) < hand_to_face_dist * sh_w = 0.25 * 0.3 = 0.075
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.48, y=0.32, visibility=1.0),  # L_WR near nose
    })
    active = classify(lm)
    assert "hand_to_face" in active, f"Expected hand_to_face, got {active}"


def test_hand_to_face_priority_over_ear():
    """Hand to face should take priority over hand to ear when both close."""
    # wrist positioned near both nose and ear
    # nose at (0.5, 0.3), left ear at (0.4, 0.28)
    # wrist at (0.44, 0.30) - close to both
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.44, y=0.30, visibility=1.0),
    })
    active = classify(lm)
    assert "hand_to_face" in active, f"Expected hand_to_face, got {active}"
    assert "hand_to_ear" not in active, f"hand_to_ear should not be active when face is closer"


def test_hand_to_ear_when_wrist_near_ear_not_above_nose():
    """Wrist near ear (not above nose) should flag hand_to_ear."""
    # dist(wrist, ear) < hand_to_ear_dist * sh_w = 0.30 * 0.3 = 0.09
    # and not above_nose
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.38, y=0.30, visibility=1.0),  # L_WR near left ear
    })
    active = classify(lm)
    assert "hand_to_ear" in active, f"Expected hand_to_ear, got {active}"


def test_hand_to_ear_not_when_above_nose():
    """Hand to ear should NOT trigger if wrist is also above nose."""
    # wrist near ear but also above nose
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.38, y=0.20, visibility=1.0),  # L_WR high and near ear
    })
    active = classify(lm)
    assert "hand_raised" in active
    assert "hand_to_ear" not in active, f"hand_to_ear should not trigger when above_nose"


def test_hand_to_ear_just_under_threshold_not_flagged():
    """Wrist just outside ear distance threshold should NOT flag hand_to_ear."""
    # ear at 0.40, threshold = 0.09, so need wrist x > 0.49
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.50, y=0.30, visibility=1.0),
    })
    active = classify(lm)
    assert "hand_to_ear" not in active, f"Expected no hand_to_ear, got {active}"


def test_look_right_when_nose_shifted_right():
    """Nose shifted right of ear midpoint should flag look_right."""
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),  # NOSE shifted right
    })
    active = classify(lm)
    assert "look_right" in active, f"Expected look_right, got {active}"


def test_look_right_just_under_threshold_not_flagged():
    """Nose just under yaw_shift should NOT flag look_right."""
    # ear_mid_x = 0.5, ear_w = 0.2, threshold = 0.45 * 0.2 = 0.09
    # need nose.x < 0.59
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.58, y=0.30, visibility=1.0),  # just under
    })
    active = classify(lm)
    assert "look_right" not in active, f"Expected no look_right, got {active}"


def test_look_left_when_nose_shifted_left():
    """Nose shifted left of ear midpoint should flag look_left."""
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.38, y=0.30, visibility=1.0),  # NOSE shifted left
    })
    active = classify(lm)
    assert "look_left" in active, f"Expected look_left, got {active}"


def test_look_left_just_under_threshold_not_flagged():
    """Nose just under yaw_shift should NOT flag look_left."""
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.42, y=0.30, visibility=1.0),  # just under
    })
    active = classify(lm)
    assert "look_left" not in active, f"Expected no look_left, got {active}"


def test_look_down_when_neck_shortened():
    """Neck ratio below threshold should flag look_down."""
    # baseline neck = 0.65, threshold = 0.55 -> need neck < 0.65 * 0.55 = 0.3575
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.42, visibility=1.0),  # NOSE lowered
    })
    active = classify(lm)
    assert "look_down" in active, f"Expected look_down, got {active}"


def test_look_down_just_under_threshold_not_flagged():
    """Neck just above threshold should NOT flag look_down."""
    # need neck >= 0.3575
    # neck = (mid_sh_y - nose.y) / sh_w = (0.5 - nose.y) / 0.3 >= 0.3575
    # nose.y <= 0.5 - 0.107 = 0.393
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.39, visibility=1.0),  # just under
    })
    active = classify(lm)
    assert "look_down" not in active, f"Expected no look_down, got {active}"


def test_leaning_when_shoulder_tilted():
    """Shoulder line tilted beyond threshold should flag leaning."""
    # baseline tilt = 0, threshold = 14 deg
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.45, visibility=1.0),  # L_SH up
        12: SimpleNamespace(x=0.65, y=0.55, visibility=1.0),  # R_SH down
    })
    active = classify(lm)
    assert "leaning" in active, f"Expected leaning, got {active}"


def test_leaning_just_under_threshold_not_flagged():
    """Tilt just under threshold should NOT flag leaning."""
    # threshold = 14 deg. tilt ~ dy/dx. dx = 0.3, need dy < 0.3*tan(14) ~ 0.075
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.48, visibility=1.0),
        12: SimpleNamespace(x=0.65, y=0.53, visibility=1.0),
    })
    active = classify(lm)
    assert "leaning" not in active, f"Expected no leaning, got {active}"


def test_low_visibility_wrist_ignored():
    """Wrist below VIS_MIN should be ignored."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=0.3),  # L_WR low visibility
    })
    active = classify(lm)
    assert "hand_raised" not in active, f"Low-vis wrist should not trigger, got {active}"


def test_two_gestures_at_once():
    """Should detect multiple simultaneous gestures."""
    # hand raised + look right
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),  # L_WR raised
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),   # NOSE right
    })
    active = classify(lm)
    assert "hand_raised" in active
    assert "look_right" in active
    assert len(active) >= 2


def test_calibrated_baseline_shifts_look_thresholds():
    """Calibrated baseline should shift look_left/right thresholds."""
    # baseline yaw = 0.2 (user naturally looks slightly right)
    base = {"yaw": 0.2, "neck": 0.65, "tilt": 0.0}
    # nose at 0.62 -> dyaw = 0.62 - 0.5 - 0.2 = -0.08? Wait...
    # ear_mid_x = 0.5, nose = 0.62 -> yaw = (0.62-0.5)/0.2 = 0.6
    # dyaw = 0.6 - 0.2 = 0.4 < 0.45 -> not flagged
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    active = classify(lm, base=base)
    assert "look_right" not in active, f"With baseline, should not flag, got {active}"

    # Now nose further right: nose at 0.67 -> yaw = 0.85 -> dyaw = 0.65 > 0.45 -> flagged
    lm2 = make_landmarks(overrides={
        0: SimpleNamespace(x=0.67, y=0.30, visibility=1.0),
    })
    active2 = classify(lm2, base=base)
    assert "look_right" in active2, f"With baseline + further shift, should flag, got {active2}"


def test_baseline_shifts_look_down():
    """Calibrated baseline neck should shift look_down threshold."""
    base = {"yaw": 0.0, "neck": 0.7, "tilt": 0.0}  # user has longer neck
    # threshold = 0.7 * 0.55 = 0.385
    # need neck < 0.385 -> (0.5 - nose.y)/0.3 < 0.385 -> nose.y > 0.3845
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.40, visibility=1.0),
    })
    active = classify(lm, base=base)
    assert "look_down" in active, f"With longer neck baseline, should flag, got {active}"


def test_baseline_shifts_leaning():
    """Calibrated baseline tilt should shift leaning threshold."""
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 5.0}  # user naturally leans 5 deg
    # threshold = 14 deg from baseline, so need tilt > 19 or < -9
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.45, visibility=1.0),
        12: SimpleNamespace(x=0.65, y=0.55, visibility=1.0),  # tilt ~ 18 deg
    })
    active = classify(lm, base=base)
    # tilt ~ 18, baseline = 5, diff = 13 < 14 -> not flagged
    assert "leaning" not in active, f"With baseline, should not flag, got {active}"

    # More tilt
    lm2 = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.40, visibility=1.0),
        12: SimpleNamespace(x=0.65, y=0.60, visibility=1.0),  # tilt ~ 33 deg
    })
    active2 = classify(lm2, base=base)
    assert "leaning" in active2, f"With more tilt, should flag, got {active2}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])