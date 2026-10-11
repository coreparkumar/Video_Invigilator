"""Window manager: composes canvas, handles resize, mouse/keyboard, toast.

Pure logic for state management; OpenCV calls for rendering only.
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable

from ui.config import (
    MIN_WINDOW_W, MIN_WINDOW_H,
    COLORS, FONT, LINE_AA,
)
from ui.layout import compute_layout, Layout, Rect, get_window_size, scale_canvas_for_window
from ui.panels import (
    PanelState, GestureProgress,
    render_video_panel, render_audio_panel, render_download_button,
    render_camera_frame, render_spectrogram, render_toast,
)
from ui.textfit import fit_text


# =============================================================================
# Window state
# =============================================================================

@dataclass
class WindowState:
    """Mutable window state."""
    # Interaction
    mouse_pos: tuple[int, int] = (-1, -1)
    mouse_down: bool = False
    button_hovered: bool = False
    button_pressed: bool = False

    # Toast
    toast_message: str = ""
    toast_color: tuple = COLORS.TOAST_SUCCESS
    toast_bg: tuple = COLORS.TOAST_BG
    toast_expires: float = 0.0

    # Window
    window_name: str = "Edge Invigilator"
    fullscreen: bool = False
    last_size: tuple[int, int] = (1280, 720)

    # Performance
    frame_times: List[float] = field(default_factory=list)
    max_frame_times: int = 60

    # Callbacks
    on_download: Optional[Callable[[], str]] = None  # returns filename or empty on fail


# =============================================================================
# Helpers
# =============================================================================

def point_in_rect(px: int, py: int, rect: Rect) -> bool:
    return rect.x <= px < rect.x + rect.w and rect.y <= py < rect.y + rect.h


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# =============================================================================
# Window manager
# =============================================================================

class WindowManager:
    """Manages the main window: layout, rendering, input, toast."""

    def __init__(
        self,
        window_name: str = "Edge Invigilator",
        initial_size: tuple[int, int] = (1280, 720),
        on_download: Optional[Callable[[], str]] = None,
    ):
        self.state = WindowState(
            window_name=window_name,
            last_size=initial_size,
            on_download=on_download,
        )
        self._layout: Optional[Layout] = None
        self._canvas: Optional[np.ndarray] = None
        self._cv_window_created = False
        self._recompute_layout(*initial_size)

    def _recompute_layout(self, w: int, h: int) -> None:
        self._layout = compute_layout(w, h)
        self._canvas = np.zeros((self._layout.window_h, self._layout.window_w, 3), dtype=np.uint8)

    def _ensure_window(self) -> None:
        if not self._cv_window_created:
            cv2.namedWindow(self.state.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.state.window_name, *self.state.last_size)
            cv2.setMouseCallback(self.state.window_name, self._on_mouse)
            self._cv_window_created = True

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        self.state.mouse_pos = (x, y)

        if self._layout is None:
            return

        btn_rect = self._layout.download_button
        hovered = point_in_rect(x, y, btn_rect)
        self.state.button_hovered = hovered

        if event == cv2.EVENT_LBUTTONDOWN and hovered:
            self.state.mouse_down = True
            self.state.button_pressed = True
        elif event == cv2.EVENT_LBUTTONUP:
            if self.state.mouse_down and hovered and self.state.on_download:
                # Trigger download
                filename = self.state.on_download()
                if filename:
                    self.show_toast(f"Saved: {filename}", COLORS.TOAST_SUCCESS, COLORS.TOAST_BG, 4.0)
                else:
                    self.show_toast("Save failed", COLORS.TOAST_ERROR, COLORS.TOAST_BG, 3.0)
            self.state.mouse_down = False
            self.state.button_pressed = False

    def show_toast(self, message: str, color: tuple = COLORS.TOAST_SUCCESS,
                   bg_color: tuple = COLORS.TOAST_BG, duration: float = 4.0) -> None:
        """Show a toast message for duration seconds."""
        self.state.toast_message = message
        self.state.toast_color = color
        self.state.toast_bg = bg_color
        self.state.toast_expires = time.time() + duration

    def handle_key(self, key: int, panel_state: PanelState) -> bool:
        """Handle key press. Returns True if key consumed."""
        now = time.time()

        if key == ord('q') or key == 27:  # ESC
            return False  # Signal to quit
        elif key == ord('f'):  # Toggle fullscreen
            self.toggle_fullscreen()
        elif key == ord('d'):  # Download findings
            if self.state.on_download:
                filename = self.state.on_download()
                if filename:
                    self.show_toast(f"Saved: {filename}", COLORS.TOAST_SUCCESS, COLORS.TOAST_BG, 4.0)
                else:
                    self.show_toast("Save failed", COLORS.TOAST_ERROR, COLORS.TOAST_BG, 3.0)
        elif key == ord('c'):  # Calibrate (handled by main loop)
            pass
        elif key == ord('n'):  # Recalibrate audio
            pass
        elif key == ord('m'):  # Mute toggle
            pass
        elif key == ord('s'):  # Skeleton toggle
            pass

        return True  # Continue

    def toggle_fullscreen(self) -> None:
        if not self._cv_window_created:
            return
        self.state.fullscreen = not self.state.fullscreen
        if self.state.fullscreen:
            cv2.setWindowProperty(self.state.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.state.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.state.window_name, *self.state.last_size)

    def update_layout(self) -> None:
        """Read real window size and recompute layout if changed."""
        if not self._cv_window_created:
            return

        w, h = get_window_size(self.state.window_name)
        if w == 0 or h == 0:
            w, h = self.state.last_size
        else:
            self.state.last_size = (w, h)

        if self._layout is None or self._layout.window_w != w or self._layout.window_h != h:
            self._recompute_layout(w, h)

    def render(
        self,
        panel_state: PanelState,
        camera_frame: Optional[np.ndarray] = None,
        landmarks: Optional[List] = None,
        connections: Optional[List] = None,
        draw_spec: Optional = None,
        spectrogram_heat: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> np.ndarray:
        """Compose and return the full canvas."""
        if now is None:
            now = time.time()

        self.update_layout()
        if self._layout is None or self._canvas is None:
            self._recompute_layout(*self.state.last_size)

        canvas = self._canvas
        layout = self._layout

        # Clear canvas
        canvas[:] = COLORS.BG_DARK

        # Left panel background
        lp = layout.left_panel
        canvas[lp.y:lp.y+lp.h, lp.x:lp.x+lp.w] = COLORS.PANEL_BG

        # Render panels on left panel area
        render_video_panel(canvas, panel_state, layout)
        render_audio_panel(canvas, panel_state, layout)

        # Download button
        render_download_button(canvas, layout, self.state.button_hovered, self.state.button_pressed)

        # Camera frame (right side)
        render_camera_frame(canvas, camera_frame, layout, landmarks, connections, draw_spec)

        # Spectrogram (bottom of camera area)
        render_spectrogram(canvas, layout, None)  # Will be passed separately

        # Toast
        if self.state.toast_message and now < self.state.toast_expires:
            render_toast(canvas, layout, self.state.toast_message,
                         self.state.toast_color, self.state.toast_bg)
        elif self.state.toast_message and now >= self.state.toast_expires:
            self.state.toast_message = ""

        # FPS overlay (top-right of camera area)
        if panel_state.flags_count >= 0:  # Always show
            ca = layout.camera_area
            fps_text = f"{1000.0 / max(1, np.mean(self.state.frame_times) * 1000):.0f} FPS" if self.state.frame_times else "-- FPS"
            fit_text(canvas[ca.y:ca.y+30, ca.x:ca.x+ca.w],
                     fps_text, (ca.w - 120, 0, 110, 28),
                     0.5, 0.2, COLORS.TEXT_MUTED, align="right")

        # Record frame time
        self.state.frame_times.append(1.0 / max(1e-6, (time.time() - now)))
        if len(self.state.frame_times) > self.state.max_frame_times:
            self.state.frame_times.pop(0)

        return canvas

    def show(self, canvas: np.ndarray) -> int:
        """Display canvas and return key code."""
        self._ensure_window()

        # Scale if window smaller than minimum
        display = canvas
        if self.state.last_size[0] < MIN_WINDOW_W or self.state.last_size[1] < MIN_WINDOW_H:
            display = scale_canvas_for_window(canvas, *self.state.last_size)

        cv2.imshow(self.state.window_name, display)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._cv_window_created:
            cv2.destroyWindow(self.state.window_name)
            self._cv_window_created = False

    def get_avg_render_ms(self) -> float:
        if not self.state.frame_times:
            return 0.0
        return np.mean(self.state.frame_times) * 1000


# =============================================================================
# Convenience: build PanelState from session/audio data
# =============================================================================

def build_panel_state(
    session_view: dict,
    audio_engine_state: Optional[dict] = None,
    audio_events_count: int = 0,
    fused_alerts: Optional[List] = None,
    last_audio_event: Optional[str] = None,
    flags_count: int = 0,
) -> PanelState:
    """Build PanelState from invigilator session and audio engine state."""
    # Face direction from session active gestures
    face_dir = "CENTER"
    for g in session_view.get("active", []):
        if g == "look_left":
            face_dir = "LEFT"
        elif g == "look_right":
            face_dir = "RIGHT"
        elif g == "look_up":
            face_dir = "UP"
        elif g == "look_down":
            face_dir = "DOWN"

    # Body movement / lean - would need additional tracking
    body_movement = "STILL"
    lean_angle = 0.0

    # Hand state
    hand_state = "NONE"
    hand_movement = "STILL"
    for g in session_view.get("active", []):
        if g == "hand_raised":
            hand_state = "RAISED"
        elif g == "hand_to_ear":
            hand_state = "AT_EAR"
        elif g == "hand_to_face":
            hand_state = "AT_FACE"

    # Gesture progress bars
    gesture_progress = []
    since = session_view.get("since", {})
    for g in session_view.get("active", []):
        from invigilator.config import GESTURES
        cfg = GESTURES.get(g)
        if cfg:
            held = time.time() - since.get(g, time.time())
            color = COLORS.FLAGGED_RED if held >= cfg.hold_seconds else COLORS.ACTIVE_AMBER
            gesture_progress.append(GestureProgress(g, cfg.label, held, cfg.hold_seconds, color))

    # Frame state
    frame_state = session_view.get("frame_state", "OUT")

    # Audio fields
    audio_label = "SILENCE"
    audio_confidence = 0.0
    level_db_above_floor = -60.0
    threshold_db = -52.0
    calibration_status = "ok"
    calibration_message = ""
    clipping = False
    muted = False
    last_audio_event = None
    last_fused_alert = None
    audio_events_count = 0

    if audio_engine_state:
        pipeline = audio_engine_state.get("pipeline", {})
        session_view = audio_engine_state.get("session", {})

        floor_db = pipeline.get("floor_mean_db", -60.0)
        thresh_db = pipeline.get("floor_threshold_db", -52.0)
        # Current level: estimate from label/confidence if not available
        # For speech/whisper, typical levels are -35 to -45 dB above floor
        label = session_view.get("last_label", "silence")
        conf = session_view.get("last_confidence", 0.0)
        if label in ("speech", "whisper"):
            # Typical speech/whisper level above floor
            level_db_above_floor = 10.0 + 20.0 * conf  # 10-30 dB above floor
        elif label == "noise_burst":
            level_db_above_floor = 5.0 + 10.0 * conf
        else:
            level_db_above_floor = 0.0
        threshold_db = thresh_db

        calibration_status = pipeline.get("calibration_status", "ok")
        calibration_message = pipeline.get("calibration_message", "")
        clipping = pipeline.get("clipping", False)
        muted = audio_engine_state.get("muted", False)

        audio_label = label.upper()
        audio_confidence = conf

        # Coverage for event
        audio_events_count = session_view.get("history_len", 0)

    if fused_alerts:
        last_fused = fused_alerts[-1]
        last_fused_alert = f"{last_fused.severity} risk={last_fused.risk:.2f} cues={last_fused.video_cues}"

    return PanelState(
        frame_state=frame_state,
        face_direction=face_dir,
        body_movement=body_movement,
        lean_angle=lean_angle,
        hand_state=hand_state,
        hand_movement=hand_movement,
        gesture_progress=gesture_progress,
        flags_count=flags_count,
        audio_label=audio_label,
        audio_confidence=audio_confidence,
        level_db_above_floor=level_db_above_floor,
        threshold_db=threshold_db,
        calibration_status=calibration_status,
        calibration_message=calibration_message,
        clipping=clipping,
        muted=muted,
        last_audio_event=last_audio_event,
        last_fused_alert=last_fused_alert,
        audio_events_count=audio_events_count,
    )


if __name__ == "__main__":
    import numpy as np
    from ui.layout import compute_layout

    print("Testing window manager...")

    layout = compute_layout(1280, 720)
    wm = WindowManager()

    # Fake panel state
    from ui.panels import PanelState, GestureProgress
    state = PanelState(
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

    # Test render
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    canvas = wm.render(state, camera_frame=frame, now=time.time())
    print(f"Canvas shape: {canvas.shape}, dtype: {canvas.dtype}")

    # Test with different sizes
    for w, h in [(960, 540), (1920, 1080), (700, 400)]:
        lm = WindowManager()
        lm._recompute_layout(w, h)
        c = lm.render(state, camera_frame=frame, now=time.time())
        assert c.shape == (max(h, MIN_WINDOW_H), max(w, MIN_WINDOW_W), 3)
        print(f"  {w}x{h} -> canvas {c.shape}")

    print("Window manager test passed!")