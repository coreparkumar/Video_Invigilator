"""Test fixtures and helpers for Edge Invigilator POC."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure repo root is on sys.path for `import audio` etc.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def make_landmarks(overrides=None):
    """
    Create 33 MediaPipe Pose landmarks for a neutral seated person.
    Returns a list of SimpleNamespace(x, y, visibility).
    overrides: dict mapping landmark index -> SimpleNamespace(x, y, visibility)
    """
    # Neutral seated pose (approximate, normalized 0-1 coordinates)
    # Origin at top-left, y increases downward
    default = {
        0:  (0.50, 0.30),   # NOSE
        7:  (0.40, 0.28),   # L_EAR
        8:  (0.60, 0.28),   # R_EAR
        11: (0.35, 0.50),   # L_SH
        12: (0.65, 0.50),   # R_SH
        15: (0.35, 0.70),   # L_WR
        16: (0.65, 0.70),   # R_WR
    }
    # Fill remaining 26 landmarks with reasonable defaults (visible, near center)
    for i in range(33):
        if i not in default:
            default[i] = (0.50, 0.40)

    lm = []
    for i in range(33):
        x, y = default[i]
        vis = 1.0
        if overrides and i in overrides:
            override = overrides[i]
            x = getattr(override, 'x', x)
            y = getattr(override, 'y', y)
            vis = getattr(override, 'visibility', vis)
        lm.append(SimpleNamespace(x=x, y=y, visibility=vis))
    return lm


class FakeClock:
    """Fake clock for testing temporal logic. Time advances only when you call advance()."""
    def __init__(self, start=1000.0):
        self._now = start

    @property
    def now(self):
        return self._now

    def advance(self, seconds):
        self._now += seconds