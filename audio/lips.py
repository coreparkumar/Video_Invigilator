"""Lip activity detection using MediaPipe Face Mesh.

Optional video cue: computes normalized lip opening variation over a window.
"""
from typing import Optional, List
import numpy as np


# MediaPipe Face Mesh landmark indices for lips
# Upper lip: 13 (top), Lower lip: 14 (bottom)
# Left corner: 61, Right corner: 291
LIP_TOP = 13
LIP_BOTTOM = 14
LIP_LEFT = 61
LIP_RIGHT = 291


class LipTracker:
    """Tracks lip opening over a sliding window to compute activity score."""

    def __init__(self, window_s: float = 1.0, hop_s: float = 0.5):
        self.window_s = window_s
        self.hop_s = hop_s
        self._openings: List[tuple[float, float]] = []  # (timestamp, opening)

    def update(self, opening: Optional[float], now: float) -> float:
        """Add a new opening measurement, return activity score 0..1."""
        if opening is not None:
            self._openings.append((now, opening))

        # Drop old entries
        cutoff = now - self.window_s
        self._openings = [(t, o) for t, o in self._openings if t >= cutoff]

        # Need at least 5 samples for reliable std
        if len(self._openings) < 5:
            return 0.0

        openings = np.array([o for _, o in self._openings])
        activity = float(np.clip(np.std(openings) / 0.05, 0.0, 1.0))
        return activity

    def clear(self) -> None:
        self._openings.clear()


def lip_opening(landmarks: list) -> Optional[float]:
    """Compute normalized lip opening from Face Mesh landmarks.

    opening = |y(top) - y(bottom)| / distance(corner_left, corner_right)

    Returns None if landmarks missing or mouth width too small.
    """
    if not landmarks or len(landmarks) < max(LIP_TOP, LIP_BOTTOM, LIP_LEFT, LIP_RIGHT) + 1:
        return None

    top = landmarks[LIP_TOP]
    bottom = landmarks[LIP_BOTTOM]
    left = landmarks[LIP_LEFT]
    right = landmarks[LIP_RIGHT]

    # Mouth width (distance between corners)
    width = np.hypot(left.x - right.x, left.y - right.y)
    if width < 1e-6:
        return None

    # Vertical opening
    opening = abs(top.y - bottom.y) / width
    return float(opening)


class FaceMeshLips:
    """MediaPipe Face Mesh wrapper for lip activity."""

    def __init__(self):
        self._mesh = None
        self._connections = None

    def _init(self):
        if self._mesh is None:
            try:
                import mediapipe as mp
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except AttributeError as e:
                raise RuntimeError(
                    "MediaPipe solutions.face_mesh not found. "
                    "Ensure mediapipe==0.10.14 is installed."
                ) from e

    def opening(self, rgb_frame: np.ndarray) -> Optional[float]:
        """Process RGB frame, return lip opening or None."""
        self._init()
        if self._mesh is None:
            return None

        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        return lip_opening(landmarks)

    def close(self) -> None:
        if self._mesh:
            self._mesh.close()
            self._mesh = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Quick smoke test with synthetic landmarks
    from types import SimpleNamespace
    import numpy as np

    print("Testing LipTracker and lip_opening...")

    # Test lip_opening with synthetic landmarks
    def make_lm(opening_ratio=0.3):
        lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(468)]
        # Corners at fixed positions
        lm[61] = SimpleNamespace(x=0.4, y=0.5, z=0.0, visibility=1.0)  # left
        lm[291] = SimpleNamespace(x=0.6, y=0.5, z=0.0, visibility=1.0)  # right
        # Top/bottom vary by opening_ratio
        lm[13] = SimpleNamespace(x=0.5, y=0.5 - opening_ratio * 0.1, z=0.0, visibility=1.0)
        lm[14] = SimpleNamespace(x=0.5, y=0.5 + opening_ratio * 0.1, z=0.0, visibility=1.0)
        return lm

    # Closed mouth
    lm_closed = make_lm(0.0)
    opening = lip_opening(lm_closed)
    print(f"  Closed mouth: {opening:.4f} (expected ~0)")

    # Talking mouth (30% opening)
    lm_talk = make_lm(0.3)
    opening = lip_opening(lm_talk)
    print(f"  Talking (30%): {opening:.4f} (expected ~0.3)")

    # Scale invariance: double all coords
    lm_talk2 = make_lm(0.3)
    for p in lm_talk2:
        p.x *= 2
        p.y *= 2
    opening2 = lip_opening(lm_talk2)
    print(f"  Scale invariance: {abs(opening - opening2) < 1e-6}")

    # Missing landmarks
    assert lip_opening([]) is None
    assert lip_opening([SimpleNamespace(x=0,y=0,z=0,visibility=1) for _ in range(100)]) is None

    # Test LipTracker
    from audio.config import HOP_S
    from audio.lips import LipTracker

    class FakeClock:
        def __init__(self, start=1000.0):
            self._now = start
        @property
        def now(self): return self._now
        def advance(self, s): self._now += s

    clock = FakeClock(1000.0)
    tracker = LipTracker(window_s=1.0, hop_s=0.5)

    # Closed mouth -> activity 0
    for i in range(10):
        clock.advance(0.05)  # 0.05s steps
        act = tracker.update(0.0, clock.now)
    assert tracker.update(0.0, clock.now) == 0.0
    print("  Closed mouth -> activity 0")

    # Talking mouth -> activity > 0.5
    clock = FakeClock(1000.0)
    tracker = LipTracker()
    for i in range(20):
        clock.advance(0.05)
        # Simulate mouth opening varying between 0.1 and 0.4
        opening = 0.1 + 0.3 * abs(np.sin(i * 0.5))
        tracker.update(opening, clock.now)
    activity = tracker.update(0.25, clock.now)
    assert activity > 0.5, f"Expected activity > 0.5, got {activity}"
    print(f"  Talking mouth -> activity {activity:.2f} (>0.5)")

    # Scale invariance of tracker
    clock = FakeClock(1000.0)
    tracker = LipTracker()
    for i in range(20):
        clock.advance(0.05)
        opening = (0.1 + 0.3 * abs(np.sin(i * 0.5))) * 2  # double scale
        tracker.update(opening, clock.now)
    activity2 = tracker.update(0.5, clock.now)
    assert abs(activity - activity2) < 0.1
    print("  Tracker scale invariant")

    print("All lip tests passed!")