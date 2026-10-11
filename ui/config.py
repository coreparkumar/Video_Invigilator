"""UI configuration: colors, fonts, layout constants. All thresholds marked UNTUNED where appropriate."""
import cv2
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Colors (BGR format for OpenCV)
# =============================================================================
@dataclass(frozen=True)
class Colors:
    # Backgrounds
    BG_DARK: Tuple[int, int, int] = (24, 24, 30)
    PANEL_BG: Tuple[int, int, int] = (38, 38, 48)
    PANEL_BORDER: Tuple[int, int, int] = (60, 60, 70)

    # Text
    TEXT_PRIMARY: Tuple[int, int, int] = (235, 235, 240)
    TEXT_SECONDARY: Tuple[int, int, int] = (180, 180, 190)
    TEXT_MUTED: Tuple[int, int, int] = (110, 110, 120)

    # Status colors
    OK_GREEN: Tuple[int, int, int] = (120, 200, 90)
    ACTIVE_AMBER: Tuple[int, int, int] = (0, 190, 255)
    FLAGGED_RED: Tuple[int, int, int] = (60, 60, 240)
    COOL_ACCENT: Tuple[int, int, int] = (230, 170, 60)

    # Button states
    BTN_IDLE: Tuple[int, int, int] = (50, 50, 65)
    BTN_HOVER: Tuple[int, int, int] = (70, 70, 90)
    BTN_PRESSED: Tuple[int, int, int] = (90, 90, 110)
    BTN_TEXT: Tuple[int, int, int] = (235, 235, 240)

    # Toast
    TOAST_BG: Tuple[int, int, int] = (40, 40, 50)
    TOAST_TEXT: Tuple[int, int, int] = (235, 235, 240)
    TOAST_ERROR: Tuple[int, int, int] = (60, 60, 240)
    TOAST_SUCCESS: Tuple[int, int, int] = (120, 200, 90)

    # Level meter
    METER_BG: Tuple[int, int, int] = (20, 20, 25)
    METER_FLOOR: Tuple[int, int, int] = (100, 100, 110)
    METER_THRESH: Tuple[int, int, int] = (0, 190, 255)
    METER_LEVEL: Tuple[int, int, int] = (0, 255, 0)

    # Spectrogram colormap (inferno-like stops)
    SPEC_BLACK: Tuple[int, int, int] = (0, 0, 0)
    SPEC_DARK_BLUE: Tuple[int, int, int] = (18, 0, 70)
    SPEC_BLUE: Tuple[int, int, int] = (70, 0, 120)
    SPEC_PURPLE: Tuple[int, int, int] = (130, 10, 130)
    SPEC_MAGENTA: Tuple[int, int, int] = (180, 50, 100)
    SPEC_RED: Tuple[int, int, int] = (220, 100, 50)
    SPEC_ORANGE: Tuple[int, int, int] = (240, 160, 20)
    SPEC_YELLOW: Tuple[int, int, int] = (252, 230, 70)
    SPEC_WHITE: Tuple[int, int, int] = (255, 255, 255)


COLORS = Colors()

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_AA = cv2.LINE_AA


# Video aspect ratio (typical webcam)
VIDEO_ASPECT = 4.0 / 3.0


# =============================================================================
# Font & Sizing
# =============================================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX if 'cv2' in globals() else 0  # filled at import time
LINE_AA = cv2.LINE_AA if 'cv2' in globals() else 16

# Base font scales (relative to panel width)
BASE_FONT_SCALE = 0.5
SMALL_FONT_SCALE = 0.4
TINY_FONT_SCALE = 0.35
TITLE_FONT_SCALE = 0.6

# Minimum window size (enforced)
MIN_WINDOW_W = 960
MIN_WINDOW_H = 540

# Left panel width: ~30% of window, clamped
LEFT_PANEL_MIN_W = 340
LEFT_PANEL_MAX_W = 460
LEFT_PANEL_RATIO = 0.30

# Padding & spacing (scaled from panel width)
PADDING_RATIO = 0.03
ROW_HEIGHT_RATIO = 0.045
SECTION_GAP_RATIO = 0.04
BUTTON_HEIGHT_RATIO = 0.08
TOAST_HEIGHT_RATIO = 0.05

# Progress bar
PROGRESS_BAR_HEIGHT_RATIO = 0.015

# Spectrogram
SPEC_MAX_HEIGHT_RATIO = 0.35


# =============================================================================
# Layout Dataclasses
# =============================================================================
@dataclass(frozen=True)
class Rect:
    """Integer rectangle: x, y, width, height."""
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)


@dataclass(frozen=True)
class Layout:
    """Precomputed layout rectangles for a given window size."""
    window_w: int
    window_h: int
    left_panel: Rect
    camera_area: Rect

    # Left panel sub-rects
    video_section: Rect
    audio_section: Rect
    download_button: Rect

    # Camera letterbox (within camera_area)
    camera_letterbox: Rect

    # Toast area (top of left panel or camera area)
    toast_area: Rect

    # Scaled values
    font_scale: float
    small_font_scale: float
    row_h: int
    padding: int


# =============================================================================
# Spectrogram Colormap (inferno-like, 256 stops)
# =============================================================================
def build_inferno_lut() -> list:
    """Build 256-color inferno-like LUT for spectrogram heat map."""
    stops = [
        (0.00, COLORS.SPEC_BLACK),
        (0.15, COLORS.SPEC_DARK_BLUE),
        (0.30, COLORS.SPEC_BLUE),
        (0.45, COLORS.SPEC_PURPLE),
        (0.60, COLORS.SPEC_MAGENTA),
        (0.70, COLORS.SPEC_RED),
        (0.80, COLORS.SPEC_ORANGE),
        (0.90, COLORS.SPEC_YELLOW),
        (1.00, COLORS.SPEC_WHITE),
    ]
    lut = []
    for i in range(256):
        t = i / 255.0
        # Find segment
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r = int(c0[2] + u * (c1[2] - c0[2]))
                g = int(c0[1] + u * (c1[1] - c0[1]))
                b = int(c0[0] + u * (c1[0] - c0[0]))
                lut.append((b, g, r))
                break
        else:
            lut.append(stops[-1][1])
    return lut


INFERNO_LUT = build_inferno_lut()


# =============================================================================
# Audio Label Colors
# =============================================================================
AUDIO_LABEL_COLORS = {
    "silence": COLORS.TEXT_MUTED,
    "speech": COLORS.OK_GREEN,
    "whisper": COLORS.ACTIVE_AMBER,
    "noise_burst": COLORS.FLAGGED_RED,
    "noise": COLORS.TEXT_SECONDARY,
    "calibrating": COLORS.ACTIVE_AMBER,
}

# Gesture state colors
GESTURE_STATE_COLORS = {
    "idle": COLORS.TEXT_MUTED,
    "active": COLORS.ACTIVE_AMBER,
    "flagged": COLORS.FLAGGED_RED,
}

# Frame state colors
FRAME_STATE_COLORS = {
    "IN": COLORS.OK_GREEN,
    "PARTIAL": COLORS.ACTIVE_AMBER,
    "OUT": COLORS.FLAGGED_RED,
}

# Movement level colors
MOVEMENT_COLORS = {
    "STILL": COLORS.TEXT_MUTED,
    "LOW": COLORS.ACTIVE_AMBER,
    "HIGH": COLORS.FLAGGED_RED,
}

# Hand state colors
HAND_STATE_COLORS = {
    "NONE": COLORS.TEXT_MUTED,
    "RAISED": COLORS.ACTIVE_AMBER,
    "AT_EAR": COLORS.ACTIVE_AMBER,
    "AT_FACE": COLORS.FLAGGED_RED,
}

if __name__ == "__main__":
    import cv2
    print("UI config loaded")
    print(f"INFERNO_LUT: {len(INFERNO_LUT)} colors")
    print(f"Min window: {MIN_WINDOW_W}x{MIN_WINDOW_H}")