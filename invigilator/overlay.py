"""Overlay renderer: draws skeleton, gesture panel, timers, banners, FPS, calibration status."""

from typing import Any
import cv2
import numpy as np

from invigilator.config import GESTURES


# Color constants (BGR)
COLOR_IDLE = (140, 140, 140)      # Grey
COLOR_ACTIVE = (0, 200, 255)      # Amber
COLOR_FLAGGED = (0, 0, 255)       # Red
COLOR_BANNER = (0, 0, 255)        # Red
COLOR_TEXT = (255, 255, 255)      # White
COLOR_PANEL_BG = (0, 0, 0)        # Black


def render(
    frame: np.ndarray,
    session_view: dict,
    fps: float,
    now: float,
) -> np.ndarray:
    """Render overlay on frame.

    Args:
        frame: BGR frame to draw on (modified in place and returned)
        session_view: Read-only snapshot from Session.get_view()
            - active: frozenset of active gesture keys
            - since: dict of gesture -> start time
            - banners: list of (expire_time, text)
            - calibrated: bool
        fps: Current frames per second
        now: Current timestamp

    Returns:
        The frame with overlay drawn
    """
    h, w = frame.shape[:2]

    # Draw semi-transparent panel on left side
    panel_h = 30 + 24 * len(GESTURES)
    panel_w = 330
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), COLOR_PANEL_BG, -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Draw per-gesture status with hold timer
    for i, (g, cfg) in enumerate(GESTURES.items()):
        label, need = cfg.label, cfg.hold_seconds
        on = g in session_view["active"]
        held = now - session_view["since"][g] if on else 0

        if on and held >= need:
            color = COLOR_FLAGGED
        elif on:
            color = COLOR_ACTIVE
        else:
            color = COLOR_IDLE

        txt = f"{label}  {held:.1f}/{need:.1f}s" if on else label
        cv2.putText(frame, txt, (10, 28 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Draw flag banners (top center, max 3)
    for j, (_, msg) in enumerate(session_view["banners"][-3:]):
        cv2.putText(frame, msg, (w // 2 - 160, 40 + 34 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BANNER, 2, cv2.LINE_AA)

    # Status line at bottom
    status = "calibrated (adaptive)" if session_view["calibrated"] else "NOT calibrated (press c)"
    cv2.putText(frame, f"{fps:.0f} FPS | {status} | c:calibrate s:skeleton q:quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    return frame


def draw_skeleton(
    frame: np.ndarray,
    landmarks: list,
    connections: list,
    drawing_spec: Any = None,
) -> None:
    """Draw MediaPipe pose skeleton on frame in-place."""
    if not landmarks:
        return

    # Convert landmarks to (x, y) tuples in pixel coordinates
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw landmark circles
    for pt in points:
        cv2.circle(frame, pt, 3, (0, 255, 255), -1)


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np
    from tests.conftest import make_landmarks

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Fake session view
    session_view = {
        "active": frozenset(["hand_raised", "look_right"]),
        "since": {"hand_raised": 1000.0, "look_right": 1000.0},
        "banners": [(1004.0, "FLAG: Hand raised")],
        "calibrated": True,
    }

    rendered = render(frame, session_view, 30.0, 1002.5)
    print(f"Rendered frame shape: {rendered.shape}")

    # Test draw_skeleton
    lm = make_landmarks()
    connections = [(0, 1), (1, 2), (2, 3), (3, 7)]  # Minimal connections
    draw_skeleton(rendered, lm, connections)
    print("Skeleton drawn")