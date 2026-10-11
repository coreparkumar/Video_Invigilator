"""Layout computation: pure geometry for window layout.

Returns integer rectangles for left panel, camera area, sections, button.
Pure functions, no camera, no I/O, no clock.
"""
import cv2
import numpy as np

from ui.config import (
    MIN_WINDOW_W, MIN_WINDOW_H,
    LEFT_PANEL_MIN_W, LEFT_PANEL_MAX_W, LEFT_PANEL_RATIO,
    PADDING_RATIO, ROW_HEIGHT_RATIO, SECTION_GAP_RATIO,
    BUTTON_HEIGHT_RATIO, TOAST_HEIGHT_RATIO,
    PROGRESS_BAR_HEIGHT_RATIO, SPEC_MAX_HEIGHT_RATIO,
    Rect, Layout, FONT, LINE_AA,
    VIDEO_ASPECT,
)


# Video aspect ratio (typical webcam)
VIDEO_ASPECT = 4.0 / 3.0


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compute_layout(window_w: int, window_h: int) -> Layout:
    """Compute layout rectangles for given window size.

    Enforces minimum window size. If window is smaller, returns layout for
    minimum size (caller should scale canvas).

    Args:
        window_w: Window width in pixels
        window_h: Window height in pixels

    Returns:
        Layout with all rectangles and scaled values.
    """
    # Enforce minimum
    w = max(window_w, MIN_WINDOW_W)
    h = max(window_h, MIN_WINDOW_H)

    # Left panel width: ~30% clamped
    left_w = int(clamp(w * LEFT_PANEL_RATIO, LEFT_PANEL_MIN_W, LEFT_PANEL_MAX_W))

    # Camera area (right side)
    cam_x = left_w
    cam_y = 0
    cam_w = w - left_w
    cam_h = h

    # Letterbox camera to VIDEO_ASPECT within camera area
    # Fit height first, then check width
    letterbox_h = cam_h
    letterbox_w = int(letterbox_h * VIDEO_ASPECT)
    if letterbox_w > cam_w:
        # Too wide, fit width instead
        letterbox_w = cam_w
        letterbox_h = int(letterbox_w / VIDEO_ASPECT)

    # Center letterbox in camera area
    lb_x = cam_x + (cam_w - letterbox_w) // 2
    lb_y = cam_y + (cam_h - letterbox_h) // 2

    # Left panel rect
    left_panel = Rect(0, 0, left_w, h)

    # Padding and row height derived from left panel width
    padding = max(8, int(left_w * PADDING_RATIO))
    row_h = max(20, int(left_w * ROW_HEIGHT_RATIO))
    section_gap = max(12, int(left_w * SECTION_GAP_RATIO))
    button_h = max(44, int(left_w * BUTTON_HEIGHT_RATIO))
    toast_h = max(30, int(left_w * TOAST_HEIGHT_RATIO))
    progress_h = max(4, int(left_w * PROGRESS_BAR_HEIGHT_RATIO))

    # Font scales
    font_scale = clamp(left_w / 800.0, 0.4, 0.7)
    small_font_scale = clamp(font_scale * 0.75, 0.3, 0.5)

    # Toast area at top of left panel
    toast_area = Rect(padding, padding, left_w - 2 * padding, toast_h)

    # Video section: from below toast to middle
    video_y = toast_area.y + toast_area.h + section_gap
    # Audio section takes bottom portion, button at very bottom
    # Split remaining height: video gets ~55%, audio ~35%, button fixed
    remaining_h = h - video_y - section_gap - button_h - padding
    video_section_h = int(remaining_h * 0.55)
    audio_y = video_y + video_section_h + section_gap
    audio_section_h = remaining_h - video_section_h

    video_section = Rect(padding, video_y, left_w - 2 * padding, video_section_h)
    audio_section = Rect(padding, audio_y, left_w - 2 * padding, audio_section_h)

    # Download button at bottom of left panel
    btn_y = h - button_h - padding
    download_button = Rect(padding, btn_y, left_w - 2 * padding, button_h)

    # Camera area rect (full right side)
    camera_area = Rect(cam_x, cam_y, cam_w, cam_h)

    # Camera letterbox rect
    camera_letterbox = Rect(lb_x, lb_y, letterbox_w, letterbox_h)

    return Layout(
        window_w=w,
        window_h=h,
        left_panel=left_panel,
        camera_area=camera_area,
        video_section=video_section,
        audio_section=audio_section,
        download_button=download_button,
        camera_letterbox=camera_letterbox,
        toast_area=toast_area,
        font_scale=font_scale,
        small_font_scale=small_font_scale,
        row_h=row_h,
        padding=padding,
    )


def scale_canvas_for_window(canvas: np.ndarray, window_w: int, window_h: int) -> np.ndarray:
    """Scale canvas to fit window while preserving aspect ratio.

    Used when window is smaller than MIN_WINDOW_W x MIN_WINDOW_H.
    """
    ch, cw = canvas.shape[:2]
    scale = min(window_w / cw, window_h / ch)
    if scale >= 1.0:
        return canvas
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    return cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_window_size(window_name: str) -> tuple[int, int]:
    """Get actual window size from OpenCV.

    Falls back to canvas size if window not found.
    """
    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect[2] > 0 and rect[3] > 0:
            return rect[2], rect[3]
    except Exception:
        pass
    return 0, 0


if __name__ == "__main__":
    print("Testing layout...")

    test_sizes = [
        (960, 540),
        (1280, 720),
        (1366, 768),
        (1920, 1080),
        (700, 400),  # below minimum
    ]

    for w, h in test_sizes:
        layout = compute_layout(w, h)
        print(f"\nWindow: {w}x{h} -> Layout: {layout.window_w}x{layout.window_h}")
        print(f"  Left panel: {layout.left_panel.w}x{layout.left_panel.h} at ({layout.left_panel.x},{layout.left_panel.y})")
        print(f"  Camera area: {layout.camera_area.w}x{layout.camera_area.h} at ({layout.camera_area.x},{layout.camera_area.y})")
        print(f"  Camera letterbox: {layout.camera_letterbox.w}x{layout.camera_letterbox.h} at ({layout.camera_letterbox.x},{layout.camera_letterbox.y})")
        print(f"  Video section: {layout.video_section.w}x{layout.video_section.h} at ({layout.video_section.x},{layout.video_section.y})")
        print(f"  Audio section: {layout.audio_section.w}x{layout.audio_section.h} at ({layout.audio_section.x},{layout.audio_section.y})")
        print(f"  Download btn: {layout.download_button.w}x{layout.download_button.h} at ({layout.download_button.x},{layout.download_button.y})")
        print(f"  Toast area: {layout.toast_area.w}x{layout.toast_area.h}")
        print(f"  Font scales: {layout.font_scale:.2f}, {layout.small_font_scale:.2f}, row_h={layout.row_h}, pad={layout.padding}")

        # Verify rectangles inside canvas
        canvas_w, canvas_h = layout.window_w, layout.window_h
        for name, rect in [
            ("left_panel", layout.left_panel),
            ("camera_area", layout.camera_area),
            ("video_section", layout.video_section),
            ("audio_section", layout.audio_section),
            ("download_button", layout.download_button),
            ("camera_letterbox", layout.camera_letterbox),
            ("toast_area", layout.toast_area),
        ]:
            assert 0 <= rect.x < canvas_w, f"{name} x out of bounds"
            assert 0 <= rect.y < canvas_h, f"{name} y out of bounds"
            assert rect.x + rect.w <= canvas_w, f"{name} right out of bounds"
            assert rect.y + rect.h <= canvas_h, f"{name} bottom out of bounds"

        # Verify no overlap between left panel sections
        # video_section and audio_section shouldn't overlap
        assert layout.video_section.y + layout.video_section.h <= layout.audio_section.y + 1, "Sections overlap"
        # download_button shouldn't overlap audio_section
        assert layout.audio_section.y + layout.audio_section.h <= layout.download_button.y + 1, "Button overlaps audio"

        # Camera letterbox inside camera area
        ca = layout.camera_area
        lb = layout.camera_letterbox
        assert ca.x <= lb.x and lb.x + lb.w <= ca.x + ca.w
        assert ca.y <= lb.y and lb.y + lb.h <= ca.y + ca.h

        # Letterbox aspect ratio
        aspect = lb.w / lb.h
        assert abs(aspect - VIDEO_ASPECT) < 0.02, f"Bad aspect: {aspect}"

    print("\nAll layout tests passed!")