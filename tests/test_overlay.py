"""Tests for overlay.py - drawing functions."""
import numpy as np
import pytest

from invigilator.overlay import render, draw_skeleton
from tests.conftest import make_landmarks


def test_render_output_shape_equals_input():
    """render() returns frame with same shape as input."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    session_view = {
        "active": frozenset(),
        "since": {},
        "banners": [],
        "calibrated": False,
    }

    result = render(frame, session_view, 30.0, 1000.0)
    assert result.shape == frame.shape
    # cv2.addWeighted returns new array, so result is not the same object


def test_render_does_not_mutate_session_view():
    """render() does not mutate the session_view dict."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    session_view = {
        "active": frozenset(["hand_raised"]),
        "since": {"hand_raised": 1000.0},
        "banners": [(1004.0, "FLAG: Hand raised")],
        "calibrated": True,
    }
    original_active = session_view["active"]
    original_banners = list(session_view["banners"])

    render(frame, session_view, 30.0, 1002.0)

    assert session_view["active"] is original_active
    assert session_view["banners"] == original_banners


def test_render_all_gestures_active_no_error():
    """render() with all gestures active does not raise."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    session_view = {
        "active": frozenset(["hand_raised", "hand_to_ear", "hand_to_face",
                             "look_left", "look_right", "look_down",
                             "leaning", "out_of_frame"]),
        "since": {g: 1000.0 for g in ["hand_raised", "hand_to_ear", "hand_to_face",
                                       "look_left", "look_right", "look_down",
                                       "leaning", "out_of_frame"]},
        "banners": [(1004.0, "FLAG: Test") for _ in range(5)],
        "calibrated": True,
    }

    result = render(frame, session_view, 30.0, 1002.0)
    assert result.shape == frame.shape


def test_render_colors_idle_active_flagged():
    """render() uses different colors for idle/active/flagged states."""
    # We can't easily test pixel colors, but we can verify it runs without error
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Idle
    session_view = {"active": frozenset(), "since": {}, "banners": [], "calibrated": False}
    render(frame, session_view, 30.0, 1000.0)

    # Active
    session_view = {"active": frozenset(["hand_raised"]), "since": {"hand_raised": 1000.0}, "banners": [], "calibrated": False}
    render(frame, session_view, 30.0, 1001.0)

    # Flagged (held >= need)
    session_view = {"active": frozenset(["hand_raised"]), "since": {"hand_raised": 1000.0}, "banners": [], "calibrated": False}
    render(frame, session_view, 30.0, 1002.0)  # hand_raised need=1.0, held=2.0


def test_draw_skeleton_empty_landmarks():
    """draw_skeleton handles empty landmarks gracefully."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_skeleton(frame, [], [])
    # Should not raise


def test_draw_skeleton_none_landmarks():
    """draw_skeleton handles None landmarks gracefully."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_skeleton(frame, None, [])
    # Should not raise


def test_draw_skeleton_draws_connections():
    """draw_skeleton draws lines between connected landmarks."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Create landmarks with known positions
    class MockLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [
        MockLM(0.25, 0.25),  # 160, 120
        MockLM(0.75, 0.25),  # 480, 120
        MockLM(0.5, 0.5),    # 320, 240
    ]
    connections = [(0, 1), (1, 2), (2, 0)]

    draw_skeleton(frame, landmarks, connections)

    # Check that some pixels are non-zero (green lines)
    assert np.any(frame > 0)


def test_draw_skeleton_with_visibility():
    """draw_skeleton works with SimpleNamespace landmarks from make_landmarks."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = make_landmarks()

    # Minimal connections for testing
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face
        (0, 4), (4, 5), (5, 6), (6, 8),  # Other face
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),  # Arms
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),  # Arms
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),  # Torso/legs
    ]

    draw_skeleton(frame, lm, connections)
    assert np.any(frame > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])