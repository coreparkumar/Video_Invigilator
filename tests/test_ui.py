"""Tests for ui/ modules: config, textfit, layout."""
import cv2
import numpy as np
import pytest

from ui.config import (
    COLORS, MIN_WINDOW_W, MIN_WINDOW_H, LEFT_PANEL_MIN_W, LEFT_PANEL_MAX_W,
    VIDEO_ASPECT, INFERNO_LUT, AUDIO_LABEL_COLORS,
    GESTURE_STATE_COLORS, FRAME_STATE_COLORS, MOVEMENT_COLORS, HAND_STATE_COLORS,
    FONT, LINE_AA,
)
from ui.textfit import fit_text, wrap_text, draw_wrapped, text_size
from ui.layout import compute_layout, scale_canvas_for_window, get_window_size


# =============================================================================
# ui/config tests
# =============================================================================

def test_config_colors():
    """Colors have expected structure."""
    assert isinstance(COLORS.BG_DARK, tuple)
    assert len(COLORS.BG_DARK) == 3
    assert all(0 <= c <= 255 for c in COLORS.BG_DARK)
    assert isinstance(COLORS.ACTIVE_AMBER, tuple)
    assert isinstance(COLORS.FLAGGED_RED, tuple)


def test_config_constants():
    """Config constants have sane values."""
    assert MIN_WINDOW_W == 960
    assert MIN_WINDOW_H == 540
    assert LEFT_PANEL_MIN_W == 340
    assert LEFT_PANEL_MAX_W == 460
    assert VIDEO_ASPECT == 4.0 / 3.0


def test_inferno_lut():
    """INFERNO_LUT has 256 colors, all valid BGR."""
    assert len(INFERNO_LUT) == 256
    for c in INFERNO_LUT:
        assert isinstance(c, tuple)
        assert len(c) == 3
        assert all(0 <= v <= 255 for v in c)


def test_label_color_maps():
    """Label color maps have expected keys."""
    assert "speech" in AUDIO_LABEL_COLORS
    assert "silence" in AUDIO_LABEL_COLORS
    assert "whisper" in AUDIO_LABEL_COLORS

    assert "idle" in GESTURE_STATE_COLORS
    assert "active" in GESTURE_STATE_COLORS
    assert "flagged" in GESTURE_STATE_COLORS

    assert "IN" in FRAME_STATE_COLORS
    assert "PARTIAL" in FRAME_STATE_COLORS
    assert "OUT" in FRAME_STATE_COLORS

    assert "STILL" in MOVEMENT_COLORS
    assert "LOW" in MOVEMENT_COLORS
    assert "HIGH" in MOVEMENT_COLORS

    assert "NONE" in HAND_STATE_COLORS
    assert "RAISED" in HAND_STATE_COLORS
    assert "AT_EAR" in HAND_STATE_COLORS
    assert "AT_FACE" in HAND_STATE_COLORS


# =============================================================================
# ui/textfit tests
# =============================================================================

def test_text_size():
    """text_size returns width, height."""
    w, h = text_size("Hello", 0.6, 1)
    assert w > 0
    assert h > 0
    # Empty string
    assert text_size("", 0.6) == (0, 0)


def test_fit_text_short():
    """Short text fits without scaling."""
    img = np.full((100, 400, 3), 30, dtype=np.uint8)
    scale = fit_text(img, "Short", (0, 0, 400, 100), 0.8, 0.2, (255, 255, 255))
    # Should draw something
    assert img[20, 10, 0] > 30
    # Scale should be close to max
    assert scale >= 0.7


def test_fit_text_long_truncated():
    """Very long text truncated with ellipsis."""
    img = np.full((100, 200, 3), 30, dtype=np.uint8)
    fit_text(img, "A" * 300, (0, 0, 200, 100), 0.8, 0.2, (255, 255, 255))
    # Check ellipsis drawn in right half of image
    # At min_scale text is small, check right portion
    right_half = img[:, 100:]
    assert np.any(right_half > 30), "Ellipsis should be drawn in right half"


def test_fit_text_unbroken_word():
    """Long unbroken word gets ellipsis."""
    img = np.full((100, 200, 3), 30, dtype=np.uint8)
    fit_text(img, "Supercalifragilisticexpialidocious", (0, 0, 200, 100), 0.8, 0.2, (255, 255, 255))
    assert np.any(img[10:30, 170:190] > 30)


def test_fit_text_numbers():
    """Numeric strings like '-123.456 dB' fit."""
    img = np.full((50, 200, 3), 30, dtype=np.uint8)
    fit_text(img, "-123.456 dB", (0, 0, 200, 50), 0.6, 0.2, (0, 255, 0))
    assert np.any(img[10:40, 10:100] > 30)


def test_fit_text_empty():
    """Empty string returns max_scale without drawing."""
    img = np.full((50, 200, 3), 30, dtype=np.uint8)
    scale = fit_text(img, "", (0, 0, 200, 50), 0.6, 0.2, (255, 255, 255))
    assert scale == 0.6
    # Nothing drawn
    assert np.all(img == 30)


def test_fit_text_sentinel_unchanged_outside():
    """Pixels outside box unchanged (sentinel test)."""
    sentinel = 123
    img = np.full((100, 200, 3), sentinel, dtype=np.uint8)
    box_img = img[10:90, 10:190]  # sub-image view of box
    fit_text(box_img, "Test", (0, 0, 180, 80), 0.6, 0.2, (255, 255, 255))
    # Check outside box unchanged
    assert img[5, 5, 0] == sentinel
    assert img[95, 195, 0] == sentinel
    assert img[5, 195, 0] == sentinel
    assert img[95, 5, 0] == sentinel


def test_fit_text_align_center():
    """Center alignment centers text."""
    img = np.full((100, 400, 3), 30, dtype=np.uint8)
    fit_text(img, "Centered", (0, 0, 400, 100), 0.8, 0.2, (255, 255, 255), align="center")
    # Text should be roughly centered - check pixels around center
    assert img[20, 190:210, 0].max() > 30 or img[20, 195, 0] > 30


def test_fit_text_align_right():
    """Right alignment puts text at right edge."""
    img = np.full((100, 400, 3), 30, dtype=np.uint8)
    fit_text(img, "Right", (0, 0, 400, 100), 0.8, 0.2, (255, 255, 255), align="right")
    # Text should be near right edge
    assert img[20, 350:380, 0].max() > 30


def test_wrap_text():
    """wrap_text splits into lines."""
    lines = wrap_text("This is a long message that should wrap", (0, 0, 200, 100), 0.5, max_lines=3)
    assert len(lines) <= 3
    assert len(lines) > 0


def test_wrap_text_max_lines():
    """wrap_text respects max_lines."""
    lines = wrap_text("One two three four five six seven eight nine ten", (0, 0, 150, 100), 0.5, max_lines=2)
    assert len(lines) == 2


def test_wrap_text_empty():
    """wrap_text handles empty string."""
    assert wrap_text("", (0, 0, 200, 100), 0.5) == []


def test_draw_wrapped():
    """draw_wrapped draws multiple lines."""
    img = np.full((150, 300, 3), 30, dtype=np.uint8)
    n = draw_wrapped(img, "Line one\nLine two\nLine three\nLine four", (10, 10, 280, 130), 0.5, (255, 255, 255), max_lines=3)
    assert n == 3


def test_draw_wrapped_max_lines():
    """draw_wrapped respects max_lines."""
    img = np.full((100, 300, 3), 30, dtype=np.uint8)
    n = draw_wrapped(img, "Line one\nLine two\nLine three", (10, 10, 280, 80), 0.5, (255, 255, 255), max_lines=2)
    assert n == 2


# =============================================================================
# ui/layout tests
# =============================================================================

def test_layout_basic_sizes():
    """Layout computes correct rects for standard sizes."""
    test_sizes = [
        (960, 540),
        (1280, 720),
        (1366, 768),
        (1920, 1080),
    ]

    for w, h in test_sizes:
        layout = compute_layout(w, h)

        # Window size enforced to minimum
        assert layout.window_w >= MIN_WINDOW_W
        assert layout.window_h >= MIN_WINDOW_H

        # Left panel width clamped
        assert LEFT_PANEL_MIN_W <= layout.left_panel.w <= LEFT_PANEL_MAX_W

        # All rects inside canvas
        cw, ch = layout.window_w, layout.window_h
        for name, rect in [
            ("left_panel", layout.left_panel),
            ("camera_area", layout.camera_area),
            ("video_section", layout.video_section),
            ("audio_section", layout.audio_section),
            ("download_button", layout.download_button),
            ("camera_letterbox", layout.camera_letterbox),
            ("toast_area", layout.toast_area),
        ]:
            assert 0 <= rect.x < cw, f"{name} x out of bounds for {w}x{h}"
            assert 0 <= rect.y < ch, f"{name} y out of bounds for {w}x{h}"
            assert rect.x + rect.w <= cw, f"{name} right out of bounds for {w}x{h}"
            assert rect.y + rect.h <= ch, f"{name} bottom out of bounds for {w}x{h}"

        # No overlap between left panel sections
        assert layout.video_section.y + layout.video_section.h <= layout.audio_section.y + 1
        assert layout.audio_section.y + layout.audio_section.h <= layout.download_button.y + 1

        # Camera letterbox inside camera area
        ca = layout.camera_area
        lb = layout.camera_letterbox
        assert ca.x <= lb.x and lb.x + lb.w <= ca.x + ca.w
        assert ca.y <= lb.y and lb.y + lb.h <= ca.y + ca.h

        # Letterbox aspect ratio preserved
        aspect = lb.w / lb.h
        assert abs(aspect - VIDEO_ASPECT) < 0.02, f"Bad aspect {aspect} for {w}x{h}"

        # Font scales in range
        assert 0.4 <= layout.font_scale <= 0.7
        assert 0.3 <= layout.small_font_scale <= 0.5
        assert layout.row_h >= 20
        assert layout.padding >= 8

        # Button height at least 44px
        assert layout.download_button.h >= 44


def test_layout_below_minimum():
    """Layout for sub-minimum window returns min-size layout."""
    layout = compute_layout(700, 400)
    assert layout.window_w == MIN_WINDOW_W
    assert layout.window_h == MIN_WINDOW_H
    # But caller should use scale_canvas_for_window


def test_scale_canvas_for_window():
    """scale_canvas_for_window downscales canvas."""
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Window smaller than canvas
    scaled = scale_canvas_for_window(canvas, 960, 540)
    assert scaled.shape[1] == 960
    assert scaled.shape[0] == 540

    # Window larger than canvas - no upscale
    canvas2 = np.zeros((540, 960, 3), dtype=np.uint8)
    scaled2 = scale_canvas_for_window(canvas2, 1920, 1080)
    assert scaled2.shape[1] == 960
    assert scaled2.shape[0] == 540


def test_get_window_size_no_window():
    """get_window_size returns (0, 0) for non-existent window."""
    w, h = get_window_size("__nonexistent_window_12345__")
    assert w == 0 and h == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])