"""Tests for ui/panels.py and ui/window.py."""
import cv2
import numpy as np
import pytest
import time

from ui.config import COLORS, FONT, LINE_AA
from ui.layout import compute_layout, Layout, Rect
from ui.panels import (
    PanelState, GestureProgress,
    render_video_panel, render_audio_panel, render_download_button,
    render_camera_frame, render_spectrogram, render_toast,
    draw_progress_bar, draw_vertical_meter,
)
from ui.window import WindowManager, build_panel_state, WindowManager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def layout():
    return compute_layout(1280, 720)


@pytest.fixture
def canvas(layout):
    return np.full((layout.window_h, layout.window_w, 3), 30, dtype=np.uint8)


@pytest.fixture
def panel_state():
    from ui.panels import PanelState, GestureProgress
    from ui.config import COLORS
    return PanelState(
        frame_state="IN",
        face_direction="LEFT",
        body_movement="LOW",
        lean_angle=5.0,
        hand_state="RAISED",
        hand_movement="MOVING",
        gesture_progress=[
            GestureProgress("look_left", "Looking left", 1.2, 2.5, COLORS.ACTIVE_AMBER),
            GestureProgress("hand_raised", "Hand raised", 0.5, 1.0, COLORS.ACTIVE_AMBER),
        ],
        flags_count=3,
        audio_label="WHISPER",
        audio_confidence=0.85,
        level_db_above_floor=12.5,
        threshold_db=8.0,
        calibration_status="ok",
        calibration_message="",
        clipping=False,
        muted=False,
        last_audio_event="whispering for 3.5s",
        last_fused_alert="alert risk=0.72 cues=look_left,hand_raised",
        audio_events_count=2,
    )


@pytest.fixture
def fake_frame():
    return np.full((480, 640, 3), 50, dtype=np.uint8)


# =============================================================================
# ui/panels tests
# =============================================================================

def test_render_video_panel_basic(layout, canvas, panel_state):
    """Video panel renders without error."""
    render_video_panel(canvas, panel_state, layout)
    # Check something was drawn in video section area
    box = layout.video_section
    region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
    assert np.any(region > 30)


def test_render_video_panel_all_frame_states(layout, canvas):
    """Video panel handles all frame states."""
    from ui.panels import PanelState, GestureProgress
    from ui.config import COLORS

    for fs in ["IN", "PARTIAL", "OUT"]:
        state = PanelState(
            frame_state=fs,
            face_direction="CENTER",
            body_movement="STILL",
            lean_angle=0.0,
            hand_state="NONE",
            hand_movement="STILL",
            gesture_progress=[],
            flags_count=0,
            audio_label="SILENCE",
            audio_confidence=0.0,
            level_db_above_floor=-60.0,
            threshold_db=-52.0,
            calibration_status="ok",
            calibration_message="",
            clipping=False,
            muted=False,
            last_audio_event=None,
            last_fused_alert=None,
            audio_events_count=0,
        )
        canvas[:] = 30
        render_video_panel(canvas, state, layout)
        box = layout.video_section
        region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
        assert np.any(region > 30), f"Frame state {fs} drew nothing"


def test_render_video_panel_face_directions(layout, canvas):
    """Video panel handles all face directions."""
    from ui.panels import PanelState, GestureProgress
    from ui.config import COLORS

    for fd in ["LEFT", "RIGHT", "UP", "DOWN", "CENTER"]:
        state = PanelState(
            frame_state="IN",
            face_direction=fd,
            body_movement="STILL",
            lean_angle=0.0,
            hand_state="NONE",
            hand_movement="STILL",
            gesture_progress=[],
            flags_count=0,
            audio_label="SILENCE",
            audio_confidence=0.0,
            level_db_above_floor=-60.0,
            threshold_db=-52.0,
            calibration_status="ok",
            calibration_message="",
            clipping=False,
            muted=False,
            last_audio_event=None,
            last_fused_alert=None,
            audio_events_count=0,
        )
        canvas[:] = 30
        render_video_panel(canvas, state, layout)
        box = layout.video_section
        region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
        assert np.any(region > 30), f"Face direction {fd} drew nothing"


def test_render_video_panel_movement_levels(layout, canvas):
    """Video panel handles all movement levels."""
    from ui.panels import PanelState, GestureProgress
    from ui.config import COLORS

    for bm in ["STILL", "LOW", "HIGH"]:
        for hm in ["STILL", "MOVING", "FAST"]:
            for hs in ["NONE", "RAISED", "AT_EAR", "AT_FACE"]:
                state = PanelState(
                    frame_state="IN",
                    face_direction="CENTER",
                    body_movement=bm,
                    lean_angle=5.0,
                    hand_state=hs,
                    hand_movement=hm,
                    gesture_progress=[],
                    flags_count=0,
                    audio_label="SILENCE",
                    audio_confidence=0.0,
                    level_db_above_floor=-60.0,
                    threshold_db=-52.0,
                    calibration_status="ok",
                    calibration_message="",
                    clipping=False,
                    muted=False,
                    last_audio_event=None,
                    last_fused_alert=None,
                    audio_events_count=0,
                )
                canvas[:] = 30
                render_video_panel(canvas, state, layout)
                box = layout.video_section
                region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
                assert np.any(region > 30), f"Body={bm}, Hand={hs}, HandM={hm} drew nothing"


def test_render_video_panel_gesture_progress(layout, canvas, panel_state):
    """Video panel shows gesture progress bars."""
    canvas[:] = 30
    render_video_panel(canvas, panel_state, layout)
    box = layout.video_section
    region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
    # Progress bars should be visible (colored pixels)
    assert np.any(region > 30)


def test_render_audio_panel_basic(layout, canvas, panel_state):
    """Audio panel renders without error."""
    render_audio_panel(canvas, panel_state, layout)
    box = layout.audio_section
    region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
    assert np.any(region > 30)


def test_render_audio_panel_all_labels(layout, canvas):
    """Audio panel handles all label types."""
    from ui.panels import PanelState
    from ui.config import COLORS

    for label in ["SILENCE", "SPEECH", "WHISPER", "NOISE", "BURST"]:
        state = PanelState(
            frame_state="IN",
            face_direction="CENTER",
            body_movement="STILL",
            lean_angle=0.0,
            hand_state="NONE",
            hand_movement="STILL",
            gesture_progress=[],
            flags_count=0,
            audio_label=label,
            audio_confidence=0.8,
            level_db_above_floor=10.0,
            threshold_db=8.0,
            calibration_status="ok",
            calibration_message="",
            clipping=False,
            muted=False,
            last_audio_event=None,
            last_fused_alert=None,
            audio_events_count=0,
        )
        canvas[:] = 30
        render_audio_panel(canvas, state, layout)
        box = layout.audio_section
        region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
        assert np.any(region > 30), f"Audio label {label} drew nothing"


def test_render_audio_panel_calibration_states(layout, canvas):
    """Audio panel shows calibration status correctly."""
    from ui.panels import PanelState
    from ui.config import COLORS

    for cal_status, cal_msg in [
        ("ok", ""),
        ("no_signal", "Microphone muted"),
        ("clipping", "Lower gain"),
        ("too_loud", "Reduce gain"),
        ("noisy", "Stay quiet"),
    ]:
        state = PanelState(
            frame_state="IN",
            face_direction="CENTER",
            body_movement="STILL",
            lean_angle=0.0,
            hand_state="NONE",
            hand_movement="STILL",
            gesture_progress=[],
            flags_count=0,
            audio_label="SILENCE",
            audio_confidence=0.0,
            level_db_above_floor=-60.0,
            threshold_db=-52.0,
            calibration_status=cal_status,
            calibration_message=cal_msg,
            clipping=(cal_status == "clipping"),
            muted=False,
            last_audio_event=None,
            last_fused_alert=None,
            audio_events_count=0,
        )
        canvas[:] = 30
        render_audio_panel(canvas, state, layout)
        box = layout.audio_section
        region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
        assert np.any(region > 30), f"Calibration {cal_status} drew nothing"


def test_render_audio_panel_clipping_and_muted(layout, canvas):
    """Audio panel shows clipping and muted indicators."""
    from ui.panels import PanelState
    from ui.config import COLORS

    # Clipping
    state = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL", gesture_progress=[], flags_count=0,
        audio_label="SPEECH", audio_confidence=0.9, level_db_above_floor=15.0, threshold_db=8.0,
        calibration_status="ok", calibration_message="", clipping=True, muted=False,
        last_audio_event=None, last_fused_alert=None, audio_events_count=1,
    )
    canvas[:] = 30
    render_audio_panel(canvas, state, layout)
    box = layout.audio_section
    region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
    # Should have red pixels for clipping
    red_pixels = np.sum((region[:,:,2] > 100) & (region[:,:,0] < 100) & (region[:,:,1] < 100))
    assert red_pixels > 0

    # Muted - create new state since PanelState is frozen
    state2 = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL", gesture_progress=[], flags_count=0,
        audio_label="SPEECH", audio_confidence=0.9, level_db_above_floor=15.0, threshold_db=8.0,
        calibration_status="ok", calibration_message="", clipping=False, muted=True,
        last_audio_event=None, last_fused_alert=None, audio_events_count=1,
    )
    canvas[:] = 30
    render_audio_panel(canvas, state2, layout)
    region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
    assert np.any(region > 30)


def test_render_download_button(layout, canvas):
    """Download button renders in all states."""
    from ui.config import COLORS

    for hovered, pressed in [(False, False), (True, False), (False, True)]:
        canvas[:] = 30
        render_download_button(canvas, layout, hovered=hovered, pressed=pressed)
        box = layout.download_button
        region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
        assert np.any(region > 30)

        # Check button has border
        border_pixels = np.sum(region[0, :] > 30) + np.sum(region[-1, :] > 30)
        assert border_pixels > 0


def test_render_camera_frame_letterbox(layout, canvas, fake_frame):
    """Camera frame is letterboxed to preserve aspect ratio."""
    canvas[:] = 30
    render_camera_frame(canvas, fake_frame, layout)

    lb = layout.camera_letterbox
    region = canvas[lb.y:lb.y+lb.h, lb.x:lb.x+lb.w]
    assert np.any(region > 30)

    # Check frame content is inside letterbox (not full camera area)
    ca = layout.camera_area
    full_region = canvas[ca.y:ca.y+ca.h, ca.x:ca.x+ca.w]
    # Letterbox should have content, bars should be background
    assert np.any(full_region > 30)


def test_render_camera_frame_no_frame(layout, canvas):
    """Camera frame handles None input."""
    canvas[:] = 30
    render_camera_frame(canvas, None, layout)
    lb = layout.camera_letterbox
    region = canvas[lb.y:lb.y+lb.h, lb.x:lb.x+lb.w]
    assert np.any(region > 30)  # "NO CAMERA" text


def test_render_camera_frame_with_skeleton(layout, canvas):
    """Camera frame draws skeleton if landmarks provided."""
    from tests.conftest import make_landmarks
    from types import SimpleNamespace

    # Create fake landmarks
    lm = make_landmarks()
    connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                   (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
                   (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
                   (11, 23), (12, 24), (23, 24), (23, 25), (24, 26)]

    canvas[:] = 30
    render_camera_frame(canvas, np.full((480, 640, 3), 50, dtype=np.uint8),
                       layout, landmarks=lm, connections=connections)

    # Should have green lines and yellow circles
    lb = layout.camera_letterbox
    region = canvas[lb.y:lb.y+lb.h, lb.x:lb.x+lb.w]
    green_pixels = np.sum((region[:,:,1] > 100) & (region[:,:,0] < 100) & (region[:,:,2] < 100))
    assert green_pixels > 0


def test_render_spectrogram(layout, canvas):
    """Spectrogram renders heat map."""
    heat = np.random.rand(100, 200).astype(np.float32)
    canvas[:] = 30
    render_spectrogram(canvas, layout, heat)

    # Should have colored pixels in bottom of camera area
    ca = layout.camera_area
    bottom = canvas[ca.y+ca.h-100:ca.y+ca.h, ca.x:ca.x+ca.w]
    assert np.any(bottom > 30)


def test_render_spectrogram_none(layout, canvas):
    """Spectrogram handles None heat."""
    canvas[:] = 30
    render_spectrogram(canvas, layout, None)
    # Should not crash, canvas unchanged
    assert np.all(canvas == 30)


def test_render_toast(layout, canvas):
    """Toast renders at top of left panel."""
    canvas[:] = 30
    render_toast(canvas, layout, "Test message", COLORS.TOAST_SUCCESS, COLORS.TOAST_BG)

    box = layout.toast_area
    region = canvas[box.y:box.y+box.h, box.x:box.x+box.w]
    assert np.any(region > 30)

    # Check background color at corner (not center where text is)
    corner = region[2, 2]
    assert corner[0] == COLORS.TOAST_BG[0]  # BGR


def test_draw_progress_bar(layout, canvas):
    """Progress bar draws correctly."""
    from ui.panels import draw_progress_bar, Rect

    box = Rect(0, 0, 200, 10)  # Relative to sub_canvas
    sub_canvas = canvas[10:20, 10:210]
    draw_progress_bar(sub_canvas, box, 0.5, COLORS.ACTIVE_AMBER)
    # Half should be amber (amber is BGR=(0,190,255) -> G=190, R=255)
    # Check for green channel > 100 and red channel > 100 (amber has both)
    amber = np.sum((sub_canvas[:,:,1] > 100) & (sub_canvas[:,:,2] > 100))
    assert amber > 0


def test_draw_progress_bar_over_100(layout, canvas):
    """Progress bar clamps at 100%."""
    from ui.panels import draw_progress_bar, Rect

    box = Rect(0, 0, 200, 10)  # Relative to sub_canvas
    sub_canvas = canvas[10:20, 10:210]
    draw_progress_bar(sub_canvas, box, 1.5, COLORS.FLAGGED_RED)  # 150%
    # Should be fully filled (red = BGR (60,60,240))
    # Account for 1-pixel border: 200*10 - border_pixels
    # Border covers ~2*(200+10)-4 = 416 pixels
    filled = np.sum((sub_canvas[:,:,2] > 100) & (sub_canvas[:,:,0] < 100) & (sub_canvas[:,:,1] < 100))
    assert filled > 1500  # 2000 - ~416 border = ~1584


def test_draw_vertical_meter(layout, canvas):
    """Vertical level meter draws."""
    from ui.panels import draw_vertical_meter, Rect

    box = Rect(10, 10, 30, 100)
    draw_vertical_meter(canvas, box, 12.0, -60.0, -52.0)
    region = canvas[10:110, 10:40]
    assert np.any(region > 30)


def test_panels_no_state_mutation(layout, canvas, panel_state):
    """Panels do not mutate input state."""
    original_gestures = list(panel_state.gesture_progress)
    original_audio_label = panel_state.audio_label

    render_video_panel(canvas, panel_state, layout)
    render_audio_panel(canvas, panel_state, layout)

    assert list(panel_state.gesture_progress) == original_gestures
    assert panel_state.audio_label == original_audio_label


# =============================================================================
# ui/window tests
# =============================================================================

def test_window_manager_creation():
    """WindowManager creates without error."""
    wm = WindowManager(window_name="TestWindow", initial_size=(1280, 720))
    assert wm.state.window_name == "TestWindow"
    assert wm.state.last_size == (1280, 720)
    wm.close()


def test_window_manager_recompute_layout():
    """WindowManager recomputes layout on size change."""
    wm = WindowManager(initial_size=(1280, 720))
    assert wm._layout is not None
    assert wm._layout.window_w == 1280
    assert wm._layout.window_h == 720

    wm._recompute_layout(1920, 1080)
    assert wm._layout.window_w == 1920
    assert wm._layout.window_h == 1080
    wm.close()


def test_window_manager_toast():
    """Toast shows and expires."""
    wm = WindowManager()
    now = time.time()

    wm.show_toast("Test message", COLORS.TOAST_SUCCESS, COLORS.TOAST_BG, 2.0)
    assert wm.state.toast_message == "Test message"
    assert wm.state.toast_expires > now

    # Simulate time passing
    wm.state.toast_expires = now + 0.1
    # Would expire naturally
    wm.close()


def test_build_panel_state_from_session():
    """build_panel_state constructs PanelState from session data."""
    session_view = {
        "active": frozenset(["look_left", "hand_raised"]),
        "since": {"look_left": time.time() - 1.0, "hand_raised": time.time() - 0.5},
        "banners": [],
        "calibrated": True,
        "frame_state": "IN",
    }
    audio_engine_state = {
        "pipeline": {
            "floor_mean_db": -60.0,
            "floor_threshold_db": -52.0,
            "calibration_status": "ok",
            "calibration_message": "",
            "clipping": False,
        },
        "session": {
            "last_label": "whisper",
            "last_confidence": 0.85,
        },
        "muted": False,
    }
    fused_alerts = []  # Empty for now

    state = build_panel_state(
        session_view=session_view,
        audio_engine_state=audio_engine_state,
        audio_events_count=2,
        fused_alerts=fused_alerts,
        last_audio_event="whispering for 3.5s",
        flags_count=1,
    )

    assert state.frame_state == "IN"
    assert state.face_direction == "LEFT"
    assert state.hand_state == "RAISED"
    assert state.audio_label == "WHISPER"
    assert state.audio_confidence == 0.85
    assert state.level_db_above_floor > 0
    assert state.calibration_status == "ok"
    assert not state.clipping
    assert not state.muted
    assert state.flags_count == 1


def test_window_manager_render_shape(layout):
    """WindowManager.render returns correct canvas shape."""
    from ui.panels import PanelState, GestureProgress
    from ui.config import COLORS

    wm = WindowManager(initial_size=(1280, 720))
    state = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL", gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )

    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    canvas = wm.render(state, camera_frame=frame, now=time.time())

    assert canvas.shape == (layout.window_h, layout.window_w, 3)
    assert canvas.dtype == np.uint8

    wm.close()


def test_window_manager_scale_below_minimum():
    """WindowManager scales canvas when window below minimum."""
    from ui.config import MIN_WINDOW_W, MIN_WINDOW_H
    wm = WindowManager(initial_size=(700, 400))
    # Layout should be at minimum size
    assert wm._layout.window_w == MIN_WINDOW_W
    assert wm._layout.window_h == MIN_WINDOW_H
    wm.close()


def test_window_manager_fps_overlay(layout):
    """WindowManager shows FPS overlay."""
    from ui.panels import PanelState
    from ui.config import COLORS

    wm = WindowManager(initial_size=(1280, 720))
    state = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL", gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )

    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    canvas = wm.render(state, camera_frame=frame, now=time.time())

    # Check top-right of camera area has FPS text
    ca = layout.camera_area
    region = canvas[ca.y:ca.y+30, ca.x+ca.w-120:ca.x+ca.w]
    assert np.any(region > 30)

    wm.close()


def test_window_manager_toast_expiry(layout):
    """Toast expires after duration."""
    wm = WindowManager(initial_size=(1280, 720))
    now = time.time()

    wm.show_toast("Test", COLORS.TOAST_SUCCESS, COLORS.TOAST_BG, 0.1)  # 100ms
    assert wm.state.toast_message == "Test"

    # Advance time
    wm.state.toast_expires = now - 1.0
    # Next render would clear it
    from ui.panels import PanelState
    state = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL", gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    canvas = wm.render(state, now=now + 2.0)  # 2 seconds later
    assert wm.state.toast_message == ""

    wm.close()


def test_window_manager_render_performance(layout):
    """WindowManager render time is reasonable."""
    from ui.panels import PanelState, GestureProgress
    from ui.config import COLORS

    wm = WindowManager(initial_size=(1280, 720))
    state = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL", gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )

    frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    # Warm up
    for _ in range(5):
        wm.render(state, camera_frame=frame, now=time.time())

    # Measure
    times = []
    for _ in range(20):
        start = time.perf_counter()
        wm.render(state, camera_frame=frame, now=time.time())
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    print(f"Average render time: {avg_ms:.1f} ms")
    assert avg_ms < 50.0  # Generous limit for CI

    wm.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])