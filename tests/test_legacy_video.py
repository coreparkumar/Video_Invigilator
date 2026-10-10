"""Tests for legacy gesture_poc.py video detection (V1-V5)."""
import sys
sys.path.insert(0, 'legacy')

from types import SimpleNamespace
from gesture_poc import classify, measures, GESTURES, THRESHOLDS, DEFAULT_BASELINE, VIS_MIN
from tests.conftest import make_landmarks


# =============================================================================
# V1: Face direction - yaw sign convention, UP detection
# =============================================================================

def test_legacy_neutral_pose_returns_empty_set():
    """Neutral seated pose should have no active gestures."""
    lm = make_landmarks()
    active = classify(lm)
    assert active == set(), f"Expected empty set, got {active}"


def test_legacy_face_left_on_mirrored_screen():
    """Nose shifted LEFT on mirrored screen -> look_left."""
    # On mirrored frame: screen-left = smaller x
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.38, y=0.30, visibility=1.0),  # NOSE left
    })
    active = classify(lm)
    assert "look_left" in active
    assert "look_right" not in active


def test_legacy_face_right_on_mirrored_screen():
    """Nose shifted RIGHT on mirrored screen -> look_right."""
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),  # NOSE right
    })
    active = classify(lm)
    assert "look_right" in active
    assert "look_left" not in active


def test_legacy_face_down_neck_shortened():
    """Neck shortened (nose lower) -> look_down."""
    # baseline neck=0.65, ratio=0.55 -> threshold=0.3575
    # neutral neck = (0.5 - 0.3) / 0.3 = 0.667
    # need neck < 0.3575 -> nose.y > 0.5 - 0.3575*0.3 = 0.393
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.42, visibility=1.0),  # NOSE lowered
    })
    active = classify(lm)
    assert "look_down" in active


def test_legacy_face_up_neck_lengthened():
    """Neck lengthened (nose higher) -> look_up (NEW - should FAIL initially)."""
    # baseline neck=0.65, need neck > baseline * neck_ratio_up
    # neutral neck = 0.667, threshold = 0.65 * 1.25 = 0.8125
    # need neck > 0.8125 -> nose.y < 0.5 - 0.8125*0.3 = 0.256
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.20, visibility=1.0),  # NOSE raised
    })
    active = classify(lm)
    assert "look_up" in active, f"Expected look_up, got {active}"


def test_legacy_face_up_just_under_threshold_not_flagged():
    """Neck just under look_up threshold should NOT flag look_up."""
    # threshold = 0.65 * 1.25 = 0.8125
    # need neck <= 0.8125 -> nose.y >= 0.256
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.26, visibility=1.0),  # just under
    })
    active = classify(lm)
    assert "look_up" not in active, f"Expected no look_up, got {active}"


def test_legacy_calibrated_baseline_shifts_look_up():
    """Calibrated baseline should shift look_up threshold."""
    # baseline neck = 0.7 (user has longer neck naturally)
    # threshold = 0.7 * 1.25 = 0.875
    base = {"yaw": 0.0, "neck": 0.7, "tilt": 0.0}
    # neutral neck = 0.667 < 0.875 -> not flagged
    lm = make_landmarks()
    active = classify(lm, base=base)
    assert "look_up" not in active

    # nose higher: nose.y = 0.22 -> neck = (0.5-0.22)/0.3 = 0.933 > 0.875 -> flagged
    lm2 = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.22, visibility=1.0),
    })
    active2 = classify(lm2, base=base)
    assert "look_up" in active2


# =============================================================================
# V3: Frame state - IN / PARTIAL / OUT
# =============================================================================

def test_legacy_session_frame_state_in_full_visibility():
    """All key landmarks visible -> frame_state = IN."""
    from gesture_poc import InvigilatorSession
    from tests.conftest import FakeClock
    
    clock = FakeClock(1000.0)
    session = InvigilatorSession(out_dir="test_evidence")
    lm = make_landmarks()
    session.calibrate(lm)
    
    session.update(lm, None, clock.now)
    # Frame state should be IN (need to check session attribute)
    # This will FAIL until we add frame_state tracking


def test_legacy_session_frame_state_partial_low_visibility_nose():
    """Nose visibility < VIS_MIN but pose detected -> frame_state = PARTIAL (NEW)."""
    from gesture_poc import InvigilatorSession
    from tests.conftest import FakeClock
    
    clock = FakeClock(1000.0)
    session = InvigilatorSession(out_dir="test_evidence")
    session.calibrate(make_landmarks())
    
    lm = make_landmarks(overrides={
        0: SimpleNamespace(x=0.50, y=0.30, visibility=0.3),  # nose low vis
    })
    session.update(lm, None, clock.now)
    # Expected: session.frame_state == "PARTIAL"
    assert session.frame_state == "PARTIAL", f"Expected PARTIAL, got {getattr(session, 'frame_state', 'MISSING')}"


def test_legacy_session_frame_state_partial_low_visibility_shoulders():
    """Shoulders visibility < VIS_MIN but pose detected -> frame_state = PARTIAL (NEW)."""
    from gesture_poc import InvigilatorSession
    from tests.conftest import FakeClock
    
    clock = FakeClock(1000.0)
    session = InvigilatorSession(out_dir="test_evidence")
    session.calibrate(make_landmarks())
    
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.50, visibility=0.3),
        12: SimpleNamespace(x=0.65, y=0.50, visibility=0.3),
    })
    session.update(lm, None, clock.now)
    assert session.frame_state == "PARTIAL", f"Expected PARTIAL, got {getattr(session, 'frame_state', 'MISSING')}"


def test_legacy_session_frame_state_out_no_landmarks():
    """No landmarks (lm=None) for > ABSENT_GRACE_S -> frame_state = OUT."""
    from gesture_poc import InvigilatorSession
    from tests.conftest import FakeClock
    from gesture_poc import ABSENT_GRACE_S, GESTURES
    
    clock = FakeClock(1000.0)
    session = InvigilatorSession(out_dir="test_evidence")
    session.calibrate(make_landmarks())
    
    # Person leaves
    clock.advance(ABSENT_GRACE_S + 0.1)
    session.update(None, None, clock.now)
    assert session.frame_state == "OUT"
    
    # Wait for hold
    clock.advance(GESTURES["out_of_frame"][1])
    session.update(None, None, clock.now)
    assert "out_of_frame" in session.active


def test_legacy_session_frame_state_out_resets_on_return():
    """Person returns -> OUT resets, state goes back to IN."""
    from gesture_poc import InvigilatorSession
    from tests.conftest import FakeClock
    from gesture_poc import ABSENT_GRACE_S
    
    clock = FakeClock(1000.0)
    session = InvigilatorSession(out_dir="test_evidence")
    session.calibrate(make_landmarks())
    
    clock.advance(ABSENT_GRACE_S + 1.0)
    session.update(None, None, clock.now)
    
    clock.advance(1.0)
    session.update(make_landmarks(), None, clock.now)
    assert session.frame_state == "IN"
    assert "out_of_frame" not in session.active


# =============================================================================
# V4: Hands - raised, at ear, at face; priority and near-miss
# =============================================================================

def test_legacy_hand_raised_above_nose():
    """Wrist above nose by threshold -> hand_raised."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),  # L_WR raised
    })
    active = classify(lm)
    assert "hand_raised" in active


def test_legacy_hand_raised_near_miss_just_below_threshold():
    """Wrist just below threshold -> NOT hand_raised."""
    # nose.y=0.30, sh_w=0.3, threshold=0.15*0.3=0.045
    # need wr.y >= 0.30 - 0.045 = 0.255
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.256, visibility=1.0),
    })
    active = classify(lm)
    assert "hand_raised" not in active


def test_legacy_hand_to_face_priority_over_ear():
    """Wrist near both nose and ear -> hand_to_face takes priority."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.44, y=0.30, visibility=1.0),  # near both
    })
    active = classify(lm)
    assert "hand_to_face" in active
    assert "hand_to_ear" not in active


def test_legacy_hand_to_ear_not_when_above_nose():
    """Hand near ear but also above nose -> hand_raised, not hand_to_ear."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.38, y=0.20, visibility=1.0),  # high and near ear
    })
    active = classify(lm)
    assert "hand_raised" in active
    assert "hand_to_ear" not in active


def test_legacy_hand_to_ear_near_miss_just_outside():
    """Wrist just outside ear distance threshold -> NOT hand_to_ear."""
    # left ear at x=0.40, threshold=0.30*0.3=0.09
    # need wrist x > 0.49 to be outside
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.50, y=0.30, visibility=1.0),
    })
    active = classify(lm)
    assert "hand_to_ear" not in active


def test_legacy_hand_low_visibility_ignored():
    """Wrist visibility < VIS_MIN -> ignored for all hand gestures."""
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=0.3),  # low vis
    })
    active = classify(lm)
    assert "hand_raised" not in active
    assert "hand_to_face" not in active
    assert "hand_to_ear" not in active


def test_legacy_both_hands_independent():
    """Left and right hand gestures detected independently."""
    # L_WR raised (hand_raised), R_WR near nose (hand_to_face)
    # nose at (0.50, 0.30), R_EAR at (0.60, 0.28)
    # R_WR at (0.52, 0.32) -> dist to nose = 0.028, dist to ear = 0.082
    # hand_to_face_dist * sh_w = 0.25 * 0.3 = 0.075 -> hand_to_face
    # hand_to_ear_dist * sh_w = 0.30 * 0.3 = 0.09 -> but hand_to_face takes priority
    lm = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),  # L_WR raised
        16: SimpleNamespace(x=0.52, y=0.32, visibility=1.0),  # R_WR near nose
    })
    active = classify(lm)
    assert "hand_raised" in active
    assert "hand_to_face" in active
    assert "hand_to_ear" not in active


# =============================================================================
# V5: Hold timers and cooldowns - tested in test_legacy_session.py
# =============================================================================


# =============================================================================
# Body movement and lean (for dashboard BODY row)
# =============================================================================

def test_legacy_leaning_shoulder_tilt():
    """Shoulder tilt beyond threshold -> leaning."""
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.45, visibility=1.0),  # L_SH up
        12: SimpleNamespace(x=0.65, y=0.55, visibility=1.0),  # R_SH down
    })
    active = classify(lm)
    assert "leaning" in active


def test_legacy_leaning_just_under_threshold():
    """Tilt just under threshold -> NOT leaning."""
    # threshold = 14 deg. tilt ~ dy/dx. dx=0.3, need dy < 0.3*tan(14) ~ 0.075
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.48, visibility=1.0),
        12: SimpleNamespace(x=0.65, y=0.53, visibility=1.0),
    })
    active = classify(lm)
    assert "leaning" not in active


def test_legacy_calibrated_baseline_shifts_leaning():
    """Calibrated baseline tilt shifts leaning threshold."""
    base = {"yaw": 0.0, "neck": 0.65, "tilt": 5.0}  # user naturally leans 5 deg
    # threshold = 14 deg from baseline, so need tilt > 19 or < -9
    lm = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.45, visibility=1.0),
        12: SimpleNamespace(x=0.65, y=0.55, visibility=1.0),  # tilt ~ 18 deg
    })
    active = classify(lm, base=base)
    assert "leaning" not in active  # 18 - 5 = 13 < 14

    lm2 = make_landmarks(overrides={
        11: SimpleNamespace(x=0.35, y=0.40, visibility=1.0),
        12: SimpleNamespace(x=0.65, y=0.60, visibility=1.0),  # tilt ~ 33 deg
    })
    active2 = classify(lm2, base=base)
    assert "leaning" in active2  # 33 - 5 = 28 > 14


# =============================================================================
# Scale invariance
# =============================================================================

def test_legacy_measures_scale_invariance():
    """Doubling all coordinates should not change yaw, neck, tilt."""
    lm1 = make_landmarks()
    m1 = measures(lm1)

    lm2 = make_landmarks()
    for p in lm2:
        p.x *= 2
        p.y *= 2
    m2 = measures(lm2)

    assert abs(m1["yaw"] - m2["yaw"]) < 1e-6
    assert abs(m1["neck"] - m2["neck"]) < 1e-6
    assert abs(m1["tilt"] - m2["tilt"]) < 1e-6


def test_legacy_classify_scale_invariance():
    """Classify results should be scale invariant."""
    lm1 = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    active1 = classify(lm1)

    # Create lm2 with SAME overrides, then double all coordinates
    lm2 = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    for p in lm2:
        p.x *= 2
        p.y *= 2
    active2 = classify(lm2)

    assert active1 == active2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])