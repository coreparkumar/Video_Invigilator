"""Panel renderers: draw VIDEO DETECTOR and AUDIO DETECTOR sections from PanelState.

Pure functions: draw on sub-image views, no camera, no I/O, no clock.
"""
from dataclasses import dataclass
from typing import Optional, List
import cv2
import numpy as np

from ui.config import (
    COLORS, FONT, LINE_AA,
    GESTURE_STATE_COLORS, FRAME_STATE_COLORS,
    MOVEMENT_COLORS, HAND_STATE_COLORS,
    AUDIO_LABEL_COLORS,
    INFERNO_LUT,
)
from ui.textfit import fit_text, draw_wrapped
from ui.layout import Rect


# =============================================================================
# Read-only state dataclasses
# =============================================================================

@dataclass(frozen=True)
class GestureProgress:
    """Progress for one active gesture."""
    key: str
    label: str
    held: float
    need: float
    color: tuple[int, int, int]  # amber or red


@dataclass(frozen=True)
class PanelState:
    """Read-only snapshot for panel rendering."""
    # Video detector fields
    frame_state: str  # "IN" / "PARTIAL" / "OUT"
    face_direction: str  # "LEFT" / "RIGHT" / "UP" / "DOWN" / "CENTER"
    body_movement: str  # "STILL" / "LOW" / "HIGH"
    lean_angle: float  # degrees from baseline
    hand_state: str  # "NONE" / "RAISED" / "AT_EAR" / "AT_FACE"
    hand_movement: str  # "STILL" / "MOVING" / "FAST"
    gesture_progress: List[GestureProgress]
    flags_count: int

    # Audio detector fields
    audio_label: str  # "SILENCE" / "SPEECH" / "WHISPER" / "NOISE" / "BURST"
    audio_confidence: float
    level_db_above_floor: float
    threshold_db: float
    calibration_status: str  # "ok" / "no_signal" / "clipping" / "too_loud" / "noisy"
    calibration_message: str
    clipping: bool
    muted: bool
    last_audio_event: Optional[str]
    last_fused_alert: Optional[str]
    audio_events_count: int


# =============================================================================
# Helper functions
# =============================================================================

def draw_section_header(img: np.ndarray, text: str, y: int, font_scale: float, color: tuple = COLORS.TEXT_PRIMARY) -> int:
    """Draw section header, return next y position."""
    h, w = img.shape[:2]
    fit_text(img, text, (4, y, w - 8, 28), font_scale, 0.25, color, align="left")
    return y + 28 + 4


def draw_key_value(img: np.ndarray, key: str, value: str, y: int, font_scale: float,
                   key_color: tuple = COLORS.TEXT_SECONDARY,
                   val_color: tuple = COLORS.TEXT_PRIMARY) -> int:
    """Draw key: value pair, return next y position."""
    h, w = img.shape[:2]
    key_w, _ = cv2.getTextSize(key, FONT, font_scale, 1)[0]
    fit_text(img, key, (4, y, key_w + 4, 22), font_scale, 0.2, key_color, align="left")
    fit_text(img, value, (4 + key_w + 4, y, w - key_w - 12, 22), font_scale, 0.2, val_color, align="left")
    return y + 22 + 2


def draw_progress_bar(img: np.ndarray, box: Rect, progress: float,
                      color: tuple = COLORS.ACTIVE_AMBER,
                      bg_color: tuple = COLORS.METER_BG) -> None:
    """Draw horizontal progress bar (0.0 to 1.0+)."""
    x, y, w, h = box.x, box.y, box.w, box.h
    # Background
    cv2.rectangle(img[y:y+h, x:x+w], (0, 0), (w, h), bg_color, -1)
    # Foreground
    fill_w = int(w * min(progress, 1.0))
    if fill_w > 0:
        cv2.rectangle(img[y:y+h, x:x+fill_w], (0, 0), (fill_w, h), color, -1)
    # Border
    cv2.rectangle(img[y:y+h, x:x+w], (0, 0), (w-1, h-1), COLORS.PANEL_BORDER, 1)


def draw_vertical_meter(img: np.ndarray, box: Rect, level_db: float,
                        floor_db: float, threshold_db: float,
                        max_db: float = 40.0) -> None:
    """Draw vertical level meter (dB above floor)."""
    x, y, w, h = box.x, box.y, box.w, box.h
    # Background
    cv2.rectangle(img[y:y+h, x:x+w], (0, 0), (w, h), COLORS.METER_BG, -1)

    # Floor line (0 dB above floor)
    floor_y = int(y + h * 0.95)
    cv2.line(img, (x, floor_y), (x + w, floor_y), COLORS.METER_FLOOR, 1)

    # Threshold line
    if threshold_db > floor_db:
        thresh_ratio = (threshold_db - floor_db) / max_db
        thresh_y = int(y + h * (1 - clamp(thresh_ratio, 0, 1)))
        cv2.line(img, (x, thresh_y), (x + w, thresh_y), COLORS.METER_THRESH, 1)

    # Current level
    level_ratio = (level_db - floor_db) / max_db
    level_y = int(y + h * (1 - clamp(level_ratio, 0, 1)))
    level_y = max(y, min(y + h, level_y))
    cv2.line(img, (x, level_y), (x + w, level_y), COLORS.METER_LEVEL, 2)

    # dB label
    fit_text(img[y:y+h, x:x+w], f"{level_db:.0f} dB", (2, 2, w-4, 18), 0.3, 0.2, COLORS.TEXT_MUTED, align="right")


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def spectrogram_to_bgr(heat: np.ndarray) -> np.ndarray:
    """Convert heat array (H x W, 0..1) to BGR using inferno LUT."""
    h, w = heat.shape
    # Scale to 0..255
    idx = (heat * 255).astype(np.uint8)
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(256):
        bgr[idx == i] = INFERNO_LUT[i]
    return bgr


# =============================================================================
# Panel renderers
# =============================================================================

def render_video_panel(img: np.ndarray, state: PanelState, layout: 'Layout') -> None:
    """Render VIDEO DETECTOR section on left panel sub-image."""
    h, w = img.shape[:2]
    box = layout.video_section
    x, y, pw, ph = box.x, box.y, box.w, box.h

    # Work on sub-image view
    panel = img[y:y+ph, x:x+pw]
    ph_actual, pw_actual = panel.shape[:2]
    if pw_actual <= 0 or ph_actual <= 0:
        return

    cy = 0
    font_s = layout.font_scale
    small_s = layout.small_font_scale
    row_h = layout.row_h
    pad = layout.padding

    # Header
    cy = draw_section_header(panel, "VIDEO DETECTOR", cy, font_s)

    # Frame state
    fs_color = FRAME_STATE_COLORS.get(state.frame_state, COLORS.TEXT_PRIMARY)
    cy = draw_key_value(panel, "Frame:", state.frame_state, cy, small_s, val_color=fs_color)

    # Face direction (with "as seen on screen" note)
    fc_color = GESTURE_STATE_COLORS.get("active" if state.face_direction != "CENTER" else "idle", COLORS.TEXT_PRIMARY)
    face_text = f"{state.face_direction} (as seen on screen)" if state.face_direction != "CENTER" else "CENTER"
    cy = draw_key_value(panel, "Face:", face_text, cy, small_s, val_color=fc_color)

    # Body movement
    bm_color = MOVEMENT_COLORS.get(state.body_movement, COLORS.TEXT_PRIMARY)
    cy = draw_key_value(panel, "Body:", f"{state.body_movement} (lean {state.lean_angle:.0f}°)", cy, small_s, val_color=bm_color)

    # Hand state + movement
    hs_color = HAND_STATE_COLORS.get(state.hand_state, COLORS.TEXT_PRIMARY)
    cy = draw_key_value(panel, "Hands:", f"{state.hand_state} / {state.hand_movement}", cy, small_s, val_color=hs_color)

    # Active gestures with hold progress bars
    if state.gesture_progress:
        cy += 4
        fit_text(panel, "Active gestures:", (pad, cy, pw_actual - 2*pad, row_h), small_s, 0.2, COLORS.TEXT_SECONDARY)
        cy += row_h

        for gp in state.gesture_progress:
            if cy + row_h + 8 > ph_actual:
                break
            # Gesture label
            fit_text(panel, f"{gp.label}: {gp.held:.1f}/{gp.need:.1f}s",
                    (pad, cy, pw_actual - 2*pad, row_h), small_s, 0.2, gp.color, align="left")
            cy += row_h
            # Progress bar
            bar_box = Rect(pad, cy, pw_actual - 2*pad, 6)
            draw_progress_bar(panel, bar_box, gp.held / gp.need, gp.color)
            cy += 10

    # Flags count
    cy += 4
    fit_text(panel, f"Flags this session: {state.flags_count}", (pad, cy, pw_actual - 2*pad, row_h), small_s, 0.2, COLORS.TEXT_SECONDARY)


def render_audio_panel(img: np.ndarray, state: PanelState, layout: 'Layout') -> None:
    """Render AUDIO DETECTOR section on left panel sub-image."""
    box = layout.audio_section
    x, y, pw, ph = box.x, box.y, box.w, box.h

    panel = img[y:y+ph, x:x+pw]
    ph_actual, pw_actual = panel.shape[:2]
    if pw_actual <= 0 or ph_actual <= 0:
        return

    cy = 0
    font_s = layout.font_scale
    small_s = layout.small_font_scale
    row_h = layout.row_h
    pad = layout.padding

    # Header
    cy = draw_section_header(panel, "AUDIO DETECTOR", cy, font_s)

    # Audio label with confidence
    label_color = AUDIO_LABEL_COLORS.get(state.audio_label.lower(), COLORS.TEXT_PRIMARY)
    label_text = f"{state.audio_label} ({state.audio_confidence:.0%})"
    cy = draw_key_value(panel, "Label:", label_text, cy, small_s, val_color=label_color)

    # Level meter (vertical on right side)
    meter_box = Rect(pw_actual - 40, cy + 2, 36, 80)
    draw_vertical_meter(panel, meter_box, state.level_db_above_floor, 0.0, state.threshold_db)
    cy += 84

    # Calibration status
    cal_color = COLORS.OK_GREEN if state.calibration_status == "ok" else COLORS.ACTIVE_AMBER
    cy = draw_key_value(panel, "Calibration:", state.calibration_status.upper(), cy, small_s, val_color=cal_color)
    if state.calibration_status != "ok" and state.calibration_message:
        draw_wrapped(panel, state.calibration_message, (pad, cy, pw_actual - 2*pad, 40), small_s, COLORS.ACTIVE_AMBER, max_lines=2)
        cy += 44

    # Clipping
    if state.clipping:
        cy = draw_key_value(panel, "CLIPPING:", "YES", cy, small_s, val_color=COLORS.FLAGGED_RED)

    # Muted
    if state.muted:
        cy = draw_key_value(panel, "MUTED:", "YES", cy, small_s, val_color=COLORS.TEXT_MUTED)

    # Last audio event
    if state.last_audio_event:
        draw_wrapped(panel, f"Last event: {state.last_audio_event}", (pad, cy, pw_actual - 2*pad, 30), small_s, COLORS.TEXT_SECONDARY, max_lines=2)
        cy += 34

    # Last fused alert
    if state.last_fused_alert:
        draw_wrapped(panel, f"Fused: {state.last_fused_alert}", (pad, cy, pw_actual - 2*pad, 30), small_s, COLORS.FLAGGED_RED, max_lines=2)
        cy += 34

    # Audio events count
    fit_text(panel, f"Audio events: {state.audio_events_count}", (pad, cy, pw_actual - 2*pad, row_h), small_s, 0.2, COLORS.TEXT_SECONDARY)


def render_download_button(img: np.ndarray, layout: 'Layout', hovered: bool = False, pressed: bool = False) -> None:
    """Render DOWNLOAD FINDINGS button."""
    box = layout.download_button
    x, y, w, h = box.x, box.y, box.w, box.h

    panel = img[y:y+h, x:x+w]
    ph, pw = panel.shape[:2]
    if ph <= 0 or pw <= 0:
        return

    # Button color based on state
    if pressed:
        color = COLORS.BTN_PRESSED
    elif hovered:
        color = COLORS.BTN_HOVER
    else:
        color = COLORS.BTN_IDLE

    cv2.rectangle(panel, (0, 0), (pw-1, ph-1), color, -1)
    cv2.rectangle(panel, (0, 0), (pw-1, ph-1), COLORS.PANEL_BORDER, 2)

    # Text
    fit_text(panel, "DOWNLOAD FINDINGS (d)", (4, 4, pw-8, ph-8), layout.font_scale, 0.25, COLORS.BTN_TEXT, align="center", valign="middle")


def render_spectrogram(img: np.ndarray, layout: 'Layout', heat: Optional[np.ndarray]) -> None:
    """Render spectrogram heatmap in camera area (bottom portion)."""
    if heat is None or heat.size == 0:
        return

    box = layout.camera_area
    x, y, w, h = box.x, box.y, box.w, box.h

    # Use bottom portion of camera area
    spec_h = int(h * 0.35)
    spec_y = y + h - spec_h
    spec_box = Rect(x, spec_y, w, spec_h)

    spec_panel = img[spec_y:spec_y+spec_h, x:x+w]
    if spec_panel.shape[0] <= 0 or spec_panel.shape[1] <= 0:
        return

    # Resize heat to fit
    heat_resized = cv2.resize(heat, (w, spec_h), interpolation=cv2.INTER_LINEAR)
    bgr = spectrogram_to_bgr(heat_resized)

    # Overlay with alpha
    cv2.addWeighted(bgr, 0.7, spec_panel, 0.3, 0, spec_panel)

    # Frequency axis labels (0-8 kHz)
    for khz in [0, 2, 4, 6, 8]:
        py = spec_y + spec_h - int((khz / 8.0) * spec_h)
        cv2.line(img, (x, py), (x + w, py), COLORS.TEXT_MUTED, 1)
        fit_text(img, f"{khz}k", (x + 2, py - 2, 40, 16), 0.3, 0.2, COLORS.TEXT_MUTED)


def render_camera_frame(img: np.ndarray, frame: np.ndarray, layout: 'Layout',
                        landmarks: Optional[List] = None,
                        connections: Optional[List] = None,
                        draw_spec: Optional = None) -> None:
    """Draw letterboxed camera frame with optional skeleton."""
    lb = layout.camera_letterbox
    x, y, w, h = lb.x, lb.y, lb.w, lb.h

    if frame is None or frame.size == 0:
        # Draw placeholder
        cv2.rectangle(img, (x, y), (x + w, y + h), COLORS.METER_BG, -1)
        fit_text(img[y:y+h, x:x+w], "NO CAMERA", (0, 0, w, h), 0.6, 0.2, COLORS.TEXT_MUTED, align="center", valign="middle")
        return

    # Resize frame to letterbox (preserving aspect ratio)
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in letterbox
    dx = (w - new_w) // 2
    dy = (h - new_h) // 2

    # Paste
    canvas = img[y:y+h, x:x+w]
    canvas[dy:dy+new_h, dx:dx+new_w] = resized

    # Draw skeleton if provided (in frame coordinates, then scale)
    if landmarks is not None and connections is not None:
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                lm1 = landmarks[start_idx]
                lm2 = landmarks[end_idx]
                if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                    p1 = (int(lm1.x * new_w) + x + dx, int(lm1.y * new_h) + y + dy)
                    p2 = (int(lm2.x * new_w) + x + dx, int(lm2.y * new_h) + y + dy)
                    cv2.line(img, p1, p2, (0, 255, 0), 2)
        for lm in landmarks:
            if lm.visibility > 0.5:
                px = int(lm.x * new_w) + x + dx
                py = int(lm.y * new_h) + y + dy
                cv2.circle(img, (px, py), 3, (0, 255, 255), -1)


def render_toast(img: np.ndarray, layout: 'Layout', message: str,
                 color: tuple = COLORS.TOAST_SUCCESS,
                 bg_color: tuple = COLORS.TOAST_BG) -> None:
    """Render toast message at top of left panel."""
    box = layout.toast_area
    x, y, w, h = box.x, box.y, box.w, box.h

    panel = img[y:y+h, x:x+w]
    if panel.size == 0:
        return

    # Background
    cv2.rectangle(panel, (0, 0), (w-1, h-1), bg_color, -1)
    cv2.rectangle(panel, (0, 0), (w-1, h-1), COLORS.PANEL_BORDER, 1)

    # Text
    fit_text(panel, message, (4, 4, w-8, h-8), 0.4, 0.25, color, align="center", valign="middle")


# =============================================================================
# Type hints for Layout (forward reference)
# =============================================================================
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ui.layout import Layout


if __name__ == "__main__":
    import numpy as np
    from ui.layout import compute_layout

    print("Testing panels...")

    layout = compute_layout(1280, 720)
    canvas = np.full((720, 1280, 3), 30, dtype=np.uint8)

    # Test video panel
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

    render_video_panel(canvas, state, layout)
    render_audio_panel(canvas, state, layout)
    render_download_button(canvas, layout, hovered=False)
    cv2.imwrite("test_panels.png", canvas)
    print("Saved test_panels.png")
    print("Panel rendering test passed!")