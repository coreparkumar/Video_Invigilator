"""Feature extraction: convert landmarks into scale-invariant measurements.

Pure functions, no I/O, no time, no dependencies beyond config and math.
"""
import math
from types import SimpleNamespace

from invigilator.config import (
    NOSE, L_EAR, R_EAR, L_SH, R_SH, L_WR, R_WR,
    THRESHOLDS,
)


def dist(a: SimpleNamespace, b: SimpleNamespace) -> float:
    """Euclidean distance between two landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def measures(lm: list[SimpleNamespace]) -> dict[str, float]:
    """Scale-invariant posture measurements from pose landmarks.

    Returns dict with: sh_w (shoulder width), yaw, neck, tilt.
    All ratios normalized by shoulder width for distance invariance.
    """
    sh_w = max(dist(lm[L_SH], lm[R_SH]), 1e-6)
    mid_sh_y = (lm[L_SH].y + lm[R_SH].y) / 2
    ear_mid_x = (lm[L_EAR].x + lm[R_EAR].x) / 2
    ear_w = max(abs(lm[L_EAR].x - lm[R_EAR].x), 1e-6)

    return {
        "sh_w": sh_w,
        # >0: nose shifted toward image-right of ear midpoint (after mirror flip)
        "yaw": (lm[NOSE].x - ear_mid_x) / ear_w,
        # neck-length proxy: nose-to-shoulder-line vertical gap in shoulder widths
        "neck": (mid_sh_y - lm[NOSE].y) / sh_w,
        # shoulder line tilt in degrees
        "tilt": math.degrees(math.atan2(
            lm[R_SH].y - lm[L_SH].y,
            abs(lm[R_SH].x - lm[L_SH].x)
        )),
    }


def near_neutral(m: dict[str, float], base: dict[str, float],
                 th: dict = THRESHOLDS, gate: float = 0.5) -> bool:
    """True if posture is well inside every limit (safe to adapt baseline)."""
    return (abs(m["yaw"] - base["yaw"]) < gate * th["yaw_shift"]
            and m["neck"] > base["neck"] * (1 - gate * (1 - th["neck_ratio"]))
            and abs(m["tilt"] - base["tilt"]) < gate * th["tilt_deg"])


if __name__ == "__main__":
    # Quick smoke test
    from tests.conftest import make_landmarks
    lm = make_landmarks()
    m = measures(lm)
    print("Neutral measures:", {k: round(v, 3) for k, v in m.items()})
    print("Near neutral:", near_neutral(m, {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}))