"""Configuration module for Edge Invigilator POC.

All tunable values live here. No magic numbers in other modules.
"""
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class GestureConfig:
    label: str
    hold_seconds: float


GESTURES: dict[str, GestureConfig] = {
    "hand_raised":   GestureConfig("Hand raised",                   1.0),
    "hand_to_ear":   GestureConfig("Hand at ear (phone/earpiece?)", 1.5),
    "hand_to_face":  GestureConfig("Hand covering face/mouth",      2.0),
    "look_left":     GestureConfig("Looking left",                  2.5),
    "look_right":    GestureConfig("Looking right",                 2.5),
    "look_up":       GestureConfig("Looking up",                    2.5),
    "look_down":     GestureConfig("Looking down",                  4.0),
    "leaning":       GestureConfig("Leaning sideways",              2.5),
    "out_of_frame":  GestureConfig("Out of frame / left seat",      3.0),
}

THRESHOLDS = {
    "hand_raised_above_nose": 0.15,
    "hand_to_face_dist":      0.25,
    "hand_to_ear_dist":       0.30,
    "yaw_shift":              0.45,
    "neck_ratio":             0.55,       # look_down: neck < baseline * this
    "neck_ratio_up":          1.25,       # look_up: neck > baseline * this (UNTUNED)
    "tilt_deg":               14.0,
}

DEFAULT_BASELINE = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}
COOLDOWN_S = 5.0
VIS_MIN = 0.5
VIS_MIN_PARTIAL = 0.3   # pose found but key landmarks below this -> PARTIAL (UNTUNED)
ABSENT_GRACE_S = 0.5
BASELINE_TAU_S = 20.0
BASELINE_GATE = 0.5
CAMERA_SCAN = range(0, 4)

NOSE = 0
L_EAR = 7
R_EAR = 8
L_SH = 11
R_SH = 12
L_WR = 15
R_WR = 16


def validate() -> None:
    """Validate all configuration values."""
    required_gestures: FrozenSet[str] = frozenset([
        "hand_raised", "hand_to_ear", "hand_to_face",
        "look_left", "look_right", "look_up", "look_down",
        "leaning", "out_of_frame",
    ])
    if set(GESTURES.keys()) != required_gestures:
        raise ValueError(f"GESTURES must have exactly these keys: {required_gestures}")

    for key, cfg in GESTURES.items():
        if cfg.hold_seconds <= 0:
            raise ValueError(f"Gesture {key} hold_seconds must be > 0")

    if not (0 < VIS_MIN <= 1):
        raise ValueError("VIS_MIN must be in (0, 1]")
    if not (0 < VIS_MIN_PARTIAL < VIS_MIN):
        raise ValueError("VIS_MIN_PARTIAL must be in (0, VIS_MIN)")
    if not (0 < BASELINE_GATE <= 1):
        raise ValueError("BASELINE_GATE must be in (0, 1]")
    if not (0 < THRESHOLDS["neck_ratio"] <= 1):
        raise ValueError("THRESHOLDS['neck_ratio'] must be in (0, 1]")
    if not (THRESHOLDS["neck_ratio_up"] > 1):
        raise ValueError("THRESHOLDS['neck_ratio_up'] must be > 1")
    if COOLDOWN_S < 0:
        raise ValueError("COOLDOWN_S must be >= 0")
    if ABSENT_GRACE_S < 0:
        raise ValueError("ABSENT_GRACE_S must be >= 0")
    if BASELINE_TAU_S <= 0:
        raise ValueError("BASELINE_TAU_S must be > 0")


if __name__ == "__main__":
    validate()
    print("Config validation passed")