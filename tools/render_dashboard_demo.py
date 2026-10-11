#!/usr/bin/env python3
"""Render dashboard demo PNGs for visual review.

Generates PNGs for various dashboard states at 1280x720 and 960x540.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from ui.config import COLORS
from ui.layout import compute_layout, Layout
from ui.panels import (
    PanelState, GestureProgress,
    render_video_panel, render_audio_panel, render_download_button,
    render_camera_frame, render_spectrogram, render_toast,
)
from ui.window import WindowManager, build_panel_state


def create_output_dir():
    """Create output directory for demo images."""
    out_dir = "demo_output"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def make_panel_state(overrides=None):
    """Create a PanelState with optional overrides."""
    base = {
        "frame_state": "IN",
        "face_direction": "CENTER",
        "body_movement": "STILL",
        "lean_angle": 0.0,
        "hand_state": "NONE",
        "hand_movement": "STILL",
        "gesture_progress": [],
        "flags_count": 0,
        "audio_label": "SILENCE",
        "audio_confidence": 0.0,
        "level_db_above_floor": -60.0,
        "threshold_db": -52.0,
        "calibration_status": "ok",
        "calibration_message": "",
        "clipping": False,
        "muted": False,
        "last_audio_event": None,
        "last_fused_alert": None,
        "audio_events_count": 0,
    }
    if overrides:
        base.update(overrides)
    return PanelState(**base)


def make_canvas(layout):
    """Create a blank canvas for the layout."""
    return np.full((layout.window_h, layout.window_w, 3), 30, dtype=np.uint8)


def render_and_save(layout, panel_state, canvas, filename, camera_frame=None):
    """Render dashboard and save to file."""
    canvas_copy = canvas.copy()
    wm = WindowManager()
    wm._layout = layout
    wm._canvas = canvas_copy

    wm.render(
        panel_state=panel_state,
        camera_frame=camera_frame,
        landmarks=None,
        connections=None,
        draw_spec=None,
        spectrogram_heat=None,
        now=time.time(),
    )
    cv2.imwrite(filename, canvas_copy)
    print(f"  Saved: {filename}")


def demo_idle(out_dir, layout, canvas):
    """Idle state - nothing active."""
    print("Rendering: idle")
    state = make_panel_state()
    render_and_save(layout, state, canvas, os.path.join(out_dir, "idle.png"))


def demo_face_left_whisper(out_dir, layout, canvas):
    """Face left with whisper heat."""
    print("Rendering: face left + whisper heat")
    state = make_panel_state({
        "face_direction": "LEFT",
        "audio_label": "WHISPER",
        "audio_confidence": 0.85,
        "level_db_above_floor": 15.0,
        "gesture_progress": [
            GestureProgress("look_left", "Looking left", 1.5, 2.5, COLORS.ACTIVE_AMBER),
        ],
    })
    # Add some spectrogram heat for whisper
    heat = np.zeros((100, 200), dtype=np.float32)
    heat[30:70, 50:150] = 0.8  # Whisper band
    canvas_copy = make_canvas(layout)
    wm = WindowManager()
    wm._layout = compute_layout(1280, 720)
    wm._canvas = np.full((720, 1280, 3), 30, dtype=np.uint8)
    canvas_copy = wm._canvas
    panel_state = make_panel_state({
        "face_direction": "LEFT",
        "audio_label": "WHISPER",
        "audio_confidence": 0.85,
        "level_db_above_floor": 15.0,
        "gesture_progress": [
            GestureProgress("look_left", "Looking left", 1.5, 2.5, COLORS.ACTIVE_AMBER),
        ],
    })
    render_and_save(layout, panel_state, canvas_copy, os.path.join(out_dir, "face_left_whisper.png"))


def demo_face_up(out_dir, layout, canvas):
    """Face up."""
    print("Rendering: face up")
    state = make_panel_state({
        "face_direction": "UP",
        "gesture_progress": [
            GestureProgress("look_up", "Looking up", 1.5, 2.5, COLORS.ACTIVE_AMBER),
        ],
    })
    render_and_save(layout, state, canvas, os.path.join(out_dir, "face_up.png"))


def demo_hand_raised_words(out_dir, layout, canvas):
    """Hand raised with transcript words."""
    print("Rendering: hand raised + words")
    state = make_panel_state({
        "hand_state": "RAISED",
        "hand_movement": "MOVING",
        "gesture_progress": [
            GestureProgress("hand_raised", "Hand raised", 1.2, 1.0, COLORS.ACTIVE_AMBER),
        ],
        "audio_label": "SPEECH",
        "audio_confidence": 0.9,
        "level_db_above_floor": 20.0,
        "last_audio_event": "speech for 4.2s",
    })
    render_and_save(layout, state, canvas, os.path.join(out_dir, "hand_raised_words.png"))


def demo_fast_body_movement(out_dir, layout, canvas):
    """Fast body movement."""
    print("Rendering: fast body movement")
    state = make_panel_state({
        "body_movement": "FAST",
        "lean_angle": 15.0,
        "gesture_progress": [
            GestureProgress("leaning", "Leaning sideways", 2.0, 2.5, COLORS.ACTIVE_AMBER),
        ],
    })
    render_and_save(layout, state, canvas, os.path.join(out_dir, "fast_body_movement.png"))


def demo_partial_frame(out_dir, layout, canvas):
    """Partial frame."""
    print("Rendering: partial frame")
    state = make_panel_state({
        "frame_state": "PARTIAL",
    })
    render_and_save(layout, state, canvas, os.path.join(out_dir, "partial_frame.png"))


def demo_out_of_frame(out_dir, layout, canvas):
    """Out of frame."""
    print("Rendering: out of frame")
    state = make_panel_state({
        "frame_state": "OUT",
        "gesture_progress": [
            GestureProgress("out_of_frame", "Out of frame / left seat", 4.0, 3.0, COLORS.FLAGGED_RED),
        ],
    })
    render_and_save(layout, state, canvas, os.path.join(out_dir, "out_of_frame.png"))


def demo_clipping_noisy_cal(out_dir, layout, canvas):
    """Clipping + noisy calibration."""
    print("Rendering: clipping + noisy calibration")
    state = make_panel_state({
        "clipping": True,
        "calibration_status": "noisy",
        "calibration_message": "Environment noisy during calibration (std=5.2 dB). Stay quiet, close windows/doors, turn off fans, and press 'n' to recalibrate.",
        "audio_label": "NOISE",
        "audio_confidence": 0.7,
        "level_db_above_floor": 25.0,
    })
    render_and_save(layout, state, canvas, os.path.join(out_dir, "clipping_noisy_cal.png"))


def demo_transcription_on(out_dir, layout, canvas):
    """Transcription on."""
    print("Rendering: transcription on")
    state = make_panel_state({
        "audio_label": "SPEECH",
        "audio_confidence": 0.95,
        "level_db_above_floor": 18.0,
        "last_audio_event": "speech for 5.2s",
    })
    render_and_save(layout, state, canvas, os.path.join(out_dir, "transcription_on.png"))


def run_demos():
    """Run all demo renders."""
    out_dir = create_output_dir()
    print(f"Output directory: {out_dir}")

    sizes = [
        (1280, 720, "720p"),
        (960, 540, "540p"),
    ]

    for w, h, suffix in sizes:
        layout = compute_layout(w, h)
        canvas = make_canvas(layout)

        demo_idle(out_dir, layout, canvas)
        demo_face_left_whisper(out_dir, layout, canvas)
        demo_face_up(out_dir, layout, canvas)
        demo_hand_raised_words(out_dir, layout, canvas)
        demo_fast_body_movement(out_dir, layout, canvas)
        demo_partial_frame(out_dir, layout, canvas)
        demo_out_of_frame(out_dir, layout, canvas)
        demo_clipping_noisy_cal(out_dir, layout, canvas)
        demo_transcription_on(out_dir, layout, canvas)

        # Rename files with suffix
        for fname in os.listdir(out_dir):
            if fname.endswith(".png") and not fname.endswith(f"_{suffix}.png"):
                src = os.path.join(out_dir, fname)
                dst = os.path.join(out_dir, fname.replace(".png", f"_{suffix}.png"))
                os.rename(src, dst)

    print(f"\nAll demos rendered to {out_dir}/")
    print("Files generated:")
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith(".png"):
            print(f"  {fname}")


if __name__ == "__main__":
    run_demos()