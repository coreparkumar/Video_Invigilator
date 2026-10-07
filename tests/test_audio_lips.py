"""Tests for audio/lips.py - lip activity detection."""
import numpy as np
import pytest

from audio.lips import (
    lip_opening, LipTracker, FaceMeshLips,
    LIP_TOP, LIP_BOTTOM, LIP_LEFT, LIP_RIGHT
)
from audio.config import HOP_S


class FakeClock:
    """Fake clock for testing temporal logic."""
    def __init__(self, start=1000.0):
        self._now = start
    @property
    def now(self):
        return self._now
    def advance(self, seconds):
        self._now += seconds


def make_landmarks(opening_ratio=0.3):
    """Create synthetic landmarks with given lip opening ratio."""
    from types import SimpleNamespace
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(468)]
    # Corners
    lm[61] = SimpleNamespace(x=0.4, y=0.5, z=0.0, visibility=1.0)
    lm[291] = SimpleNamespace(x=0.6, y=0.5, z=0.0, visibility=1.0)
    # Top/bottom
    lm[13] = SimpleNamespace(x=0.5, y=0.5 - opening_ratio * 0.1, z=0.0, visibility=1.0)
    lm[14] = SimpleNamespace(x=0.5, y=0.5 + opening_ratio * 0.1, z=0.0, visibility=1.0)
    return lm


def test_lip_opening_closed():
    """Closed mouth (opening_ratio=0) -> opening ~0."""
    lm = make_landmarks(0.0)
    opening = lip_opening(lm)
    assert opening is not None
    assert abs(opening) < 0.01


def test_lip_opening_talking():
    """30% opening -> ~0.3."""
    lm = make_landmarks(0.3)
    opening = lip_opening(lm)
    assert opening is not None
    assert 0.25 <= opening <= 0.35


def test_lip_opening_scale_invariance():
    """Doubling all coordinates doesn't change opening."""
    lm1 = make_landmarks(0.3)
    o1 = lip_opening(lm1)

    lm2 = make_landmarks(0.3)
    for p in lm2:
        p.x *= 2
        p.y *= 2
    o2 = lip_opening(lm2)

    assert abs(o1 - o2) < 1e-6


def test_lip_opening_missing_landmarks():
    """Missing landmarks -> None."""
    assert lip_opening([]) is None
    from types import SimpleNamespace
    short = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(100)]
    assert lip_opening(short) is None


def test_lip_opening_zero_width():
    """Zero mouth width -> None."""
    from types import SimpleNamespace
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(468)]
    lm[61] = SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0)
    lm[291] = SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0)
    lm[13] = SimpleNamespace(x=0.5, y=0.45, z=0.0, visibility=1.0)
    lm[14] = SimpleNamespace(x=0.5, y=0.55, z=0.0, visibility=1.0)
    assert lip_opening(lm) is None


def test_landmark_indices():
    """Landmark indices are correct constants."""
    assert LIP_TOP == 13
    assert LIP_BOTTOM == 14
    assert LIP_LEFT == 61
    assert LIP_RIGHT == 291


def test_lip_tracker_closed_mouth():
    """Closed mouth (opening=0) -> activity 0."""
    from audio.config import HOP_S
    from audio.lips import LipTracker

    clock = FakeClock(1000.0)
    tracker = LipTracker(window_s=1.0, hop_s=0.5)

    for i in range(10):
        clock.advance(0.05)
        act = tracker.update(0.0, clock.now)

    assert tracker.update(0.0, clock.now) == 0.0


def test_lip_tracker_talking_mouth():
    """Talking mouth (varying opening) -> activity > 0.5."""
    from audio.lips import LipTracker

    clock = FakeClock(1000.0)
    tracker = LipTracker(window_s=1.0, hop_s=0.5)

    for i in range(20):
        clock.advance(0.05)
        opening = 0.1 + 0.3 * abs(np.sin(i * 0.5))
        tracker.update(opening, clock.now)

    activity = tracker.update(0.25, clock.now)
    assert activity > 0.5


def test_lip_tracker_scale_invariant():
    """LipTracker activity is scale invariant."""
    clock = FakeClock(1000.0)
    tracker1 = LipTracker(window_s=1.0, hop_s=0.5)

    for i in range(20):
        clock.advance(0.05)
        opening = 0.1 + 0.3 * abs(np.sin(i * 0.5))
        tracker1.update(opening, clock.now)
    act1 = tracker1.update(0.25, clock.now)

    clock = FakeClock(1000.0)
    tracker2 = LipTracker(window_s=1.0, hop_s=0.5)
    for i in range(20):
        clock.advance(0.05)
        opening = (0.1 + 0.3 * abs(np.sin(i * 0.5))) * 2  # double
        tracker2.update(opening, clock.now)
    act2 = tracker2.update(0.5, clock.now)

    assert abs(act1 - act2) < 0.1


def test_lip_tracker_insufficient_samples():
    """Fewer than 5 samples -> activity 0."""
    from audio.lips import LipTracker

    clock = FakeClock(1000.0)
    tracker = LipTracker(window_s=1.0, hop_s=0.5)

    for i in range(3):
        clock.advance(0.05)
        tracker.update(0.2, clock.now)

    assert tracker.update(0.2, clock.now) == 0.0


def test_lip_tracker_clear():
    """clear() resets tracker."""
    from audio.lips import LipTracker

    clock = FakeClock(1000.0)
    tracker = LipTracker()

    clock.advance(0.05)
    tracker.update(0.3, clock.now)
    tracker.clear()

    assert tracker.update(0.0, clock.now) == 0.0


def test_lip_opening_none_cases():
    """lip_opening returns None for various missing data cases."""
    from types import SimpleNamespace

    # Empty list
    assert lip_opening([]) is None

    # Too few landmarks
    short = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(100)]
    assert lip_opening(short) is None

    # Landmarks missing required indices (just return None gracefully)
    # This is handled by the index check in lip_opening


def test_facemesh_lips_init_error():
    """FaceMeshLips raises helpful error if mp.solutions missing."""
    from audio.lips import FaceMeshLips
    from unittest.mock import patch, MagicMock

    # Patch mediapipe.solutions to be missing
    with patch('mediapipe.solutions', None):
        lips = FaceMeshLips()
        with pytest.raises(RuntimeError, match="mediapipe==0.10.14"):
            lips.opening(np.zeros((240, 320, 3), dtype=np.uint8))


def test_facemesh_lips_context_manager():
    """FaceMeshLips works as context manager."""
    from audio.lips import FaceMeshLips
    from unittest.mock import patch, MagicMock

    with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh:
        mock_mesh = MagicMock()
        mock_face_mesh.return_value = MagicMock()

        with FaceMeshLips() as lips:
            # FaceMesh should be initialized
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])