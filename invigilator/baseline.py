"""Calibration and adaptive baseline for user-specific posture.

Baseline captures neutral posture on key press, then slowly follows natural drift
without learning away real violations.
"""
import math
from typing import Optional

from invigilator.config import BASELINE_TAU_S, BASELINE_GATE
from invigilator.features import measures, near_neutral


class Baseline:
    """Manages calibrated baseline with exponential moving average adaptation."""

    def __init__(self):
        self._base: Optional[dict[str, float]] = None  # yaw, neck, tilt
        self._uncalibrated = True

    @property
    def base(self) -> Optional[dict[str, float]]:
        return self._base

    @property
    def is_calibrated(self) -> bool:
        return self._base is not None

    def calibrate(self, lm: list) -> dict[str, float]:
        """Capture current posture as new baseline (user presses 'c')."""
        m = measures(lm)
        self._base = {k: v for k, v in m.items() if k != "sh_w"}
        self._uncalibrated = False
        return self._base

    def adapt(self, m: dict[str, float], dt: float) -> None:
        """Exponential moving average toward current posture.

        Only adapts when:
        1. Baseline is calibrated
        2. Current posture is near_neutral (within BASELINE_GATE of thresholds)
        """
        if not self.can_adapt(m):
            return

        alpha = 1.0 - math.exp(-dt / BASELINE_TAU_S)
        for k in self._base:
            self._base[k] += alpha * (m[k] - self._base[k])

    def can_adapt(self, m: dict[str, float]) -> bool:
        """Check if adaptation should happen (near neutral + calibrated)."""
        if self._base is None:
            return False
        return near_neutral(m, self._base)

    def reset(self) -> None:
        """Reset baseline to uncalibrated state."""
        self._base = None
        self._uncalibrated = True

    def get_default(self) -> dict[str, float]:
        """Return default baseline (used before calibration)."""
        from invigilator.config import DEFAULT_BASELINE
        return DEFAULT_BASELINE


if __name__ == "__main__":
    # Quick smoke test
    from tests.conftest import make_landmarks, FakeClock

    clock = FakeClock(1000.0)
    baseline = Baseline()

    # Before calibration
    print("Before calibration:", baseline.base, "is_calibrated:", baseline.is_calibrated)
    print("Default:", baseline.get_default())

    # Calibrate
    lm = make_landmarks()
    base = baseline.calibrate(lm)
    print("After calibration:", {k: round(v, 3) for k, v in base.items()})

    # Simulate small drift over time
    for i in range(10):
        clock.advance(10.0)  # 10 seconds
        lm_drift = make_landmarks(overrides={
            0: type('L', (), {'x': 0.51 + i*0.001, 'y': 0.30, 'visibility': 1.0})(),
        })
        m = measures(lm_drift)
        if baseline.can_adapt(m):
            baseline.adapt(m, 10.0)
            print(f"t={clock.now}: adapted yaw={baseline.base['yaw']:.4f}")