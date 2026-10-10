"""Gesture classifier: map measurements + baseline to active gesture set for one frame.

Pure functions, no I/O, no time, no dependencies beyond config and features.
"""
from types import SimpleNamespace

from invigilator.config import (
    GESTURES,
    THRESHOLDS,
    DEFAULT_BASELINE,
    VIS_MIN,
    NOSE, L_EAR, R_EAR, L_SH, R_SH, L_WR, R_WR,
)
from invigilator.features import measures, dist


def classify(lm: list[SimpleNamespace],
             base: dict[str, float] | None = None,
             th: dict = THRESHOLDS) -> set[str]:
    """Return the set of gesture keys currently active for one frame.

    Args:
        lm: 33 MediaPipe Pose landmarks
        base: calibrated baseline (yaw, neck, tilt). None = use DEFAULT_BASELINE
        th: thresholds dict (uses module-level THRESHOLDS by default)

    Returns:
        Set of active gesture keys from GESTURES.
    """
    active = set()
    m = measures(lm)
    b = base or DEFAULT_BASELINE
    nose = lm[NOSE]
    sh_w = m["sh_w"]
    ears = [lm[L_EAR], lm[R_EAR]]

    # Hand gestures: check each wrist
    for wr in (lm[L_WR], lm[R_WR]):
        if wr.visibility < VIS_MIN:
            continue

        # Hand raised: wrist above nose by threshold * shoulder_width
        above_nose = wr.y < nose.y - th["hand_raised_above_nose"] * sh_w
        if above_nose:
            active.add("hand_raised")

        # Hand to face: wrist close to nose (takes priority over ear)
        if dist(wr, nose) < th["hand_to_face_dist"] * sh_w:
            active.add("hand_to_face")
        # Hand to ear: wrist close to ear AND not above nose
        else:
            d_ear = min(dist(wr, e) for e in ears)
            if d_ear < th["hand_to_ear_dist"] * sh_w and not above_nose:
                active.add("hand_to_ear")

    # Head direction: yaw deviation from baseline
    dyaw = m["yaw"] - b["yaw"]
    if dyaw > th["yaw_shift"]:
        active.add("look_right")
    elif dyaw < -th["yaw_shift"]:
        active.add("look_left")

    # Looking down: neck ratio below threshold
    if m["neck"] < b["neck"] * th["neck_ratio"]:
        active.add("look_down")

    # Looking up: neck ratio above threshold
    if m["neck"] > b["neck"] * th["neck_ratio_up"]:
        active.add("look_up")

    # Leaning: shoulder tilt deviation from baseline
    if abs(m["tilt"] - b["tilt"]) > th["tilt_deg"]:
        active.add("leaning")

    return active


if __name__ == "__main__":
    # Quick smoke test
    from tests.conftest import make_landmarks
    lm = make_landmarks()
    print("Neutral:", classify(lm))

    # Hand raised
    lm_raised = make_landmarks(overrides={
        15: SimpleNamespace(x=0.35, y=0.20, visibility=1.0),
    })
    print("Hand raised:", classify(lm_raised))

    # Look right
    lm_right = make_landmarks(overrides={
        0: SimpleNamespace(x=0.62, y=0.30, visibility=1.0),
    })
    print("Look right:", classify(lm_right))