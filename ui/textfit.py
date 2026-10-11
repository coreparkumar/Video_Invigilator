"""Text fitting utilities: fit text into a box by scaling font and truncating with ellipsis.

Pure functions, OpenCV + NumPy only. No camera, no I/O, no clock.
"""
import cv2
import numpy as np

from ui.config import FONT, LINE_AA


def text_size(text: str, font_scale: float, thickness: int = 1) -> tuple[int, int]:
    """Return (width, height) of text at given font scale."""
    if not text:
        return 0, 0
    (w, h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    return w, h


def fit_text(
    img: np.ndarray,
    text: str,
    box: tuple[int, int, int, int],
    max_scale: float,
    min_scale: float = 0.25,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    align: str = "left",
    valign: str = "top",
) -> float:
    """Draw text fitted into a box.

    Shrinks font until text fits box width. If still too wide at min_scale,
    truncates with ellipsis ("...").

    Args:
        img: Destination image (modified in place). Should be a sub-image view of the box.
        text: Text to draw.
        box: (x, y, w, h) in img coordinates (relative to img top-left).
        max_scale: Starting font scale.
        min_scale: Minimum font scale before truncation.
        color: BGR color.
        thickness: Line thickness.
        align: "left", "center", or "right".
        valign: "top", "middle", or "bottom".

    Returns:
        The font scale actually used.
    """
    if not text:
        return max_scale

    x, y, w, h = box
    if w <= 0 or h <= 0:
        return max_scale

    # Ensure we're drawing on the box sub-image
    # img is expected to be the box region already
    scale = max_scale
    text_w, text_h = text_size(text, scale, thickness)

    # Shrink until fits width
    while text_w > w - 4 and scale > min_scale:  # 2px margin each side
        scale -= 0.02
        text_w, text_h = text_size(text, scale, thickness)

    # If still too wide at min_scale, truncate with ellipsis
    if text_w > w - 4:
        ellipsis = "..."
        ellipsis_w, _ = text_size(ellipsis, min_scale, thickness)
        # Binary search for max chars that fit
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = text[:mid] + ellipsis
            cw, _ = text_size(candidate, min_scale, thickness)
            if cw <= w - 4:
                lo = mid
            else:
                hi = mid - 1
        text = text[:lo] + ellipsis
        scale = min_scale
        text_w, text_h = text_size(text, scale, thickness)

    # Compute position
    if align == "center":
        tx = (w - text_w) // 2
    elif align == "right":
        tx = w - text_w - 2
    else:
        tx = 2

    if valign == "middle":
        ty = (h + text_h) // 2
    elif valign == "bottom":
        ty = h - 2
    else:
        ty = text_h + 2

    # Ensure text is drawn within image bounds
    tx = max(0, min(tx, w - text_w - 2))
    ty = max(text_h + 2, min(ty, h - 2))

    # Draw (clipped to img bounds by OpenCV)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, LINE_AA)
    return scale


def wrap_text(
    text: str,
    box: tuple[int, int, int, int],
    font_scale: float,
    thickness: int = 1,
    max_lines: int = 3,
    color: tuple[int, int, int] = (255, 255, 255),
    line_spacing: float = 1.2,
) -> list[str]:
    """Wrap text to fit box width, return list of lines (max max_lines).

    Handles explicit newlines in text. Does NOT draw.
    """
    if not text:
        return []

    _, _, w, _ = box
    if w <= 0:
        return []

    # Split by explicit newlines first
    paragraphs = text.split('\n')
    lines = []

    for para in paragraphs:
        if len(lines) >= max_lines:
            break
        words = para.split()
        current = ""
        for word in words:
            test = current + (" " if current else "") + word
            tw, _ = text_size(test, font_scale, thickness)
            if tw <= w - 4:
                current = test
            else:
                if current:
                    lines.append(current)
                    if len(lines) >= max_lines:
                        break
                current = word
        if current and len(lines) < max_lines:
            lines.append(current)
        # Add empty line between paragraphs if not last
        if len(lines) < max_lines and para != paragraphs[-1]:
            lines.append("")

    # If we have more content but hit max_lines, truncate last line with ellipsis
    if len(lines) >= max_lines:
        # Join remaining paragraphs to check if there's more content
        remaining = '\n'.join(paragraphs[len(lines):]) if len(paragraphs) > len(lines) else ""
        if remaining.strip():
            last = lines[-1]
            while len(last) > 0:
                test = last + "..."
                tw, _ = text_size(test, font_scale, thickness)
                if tw <= w - 4:
                    lines[-1] = test
                    break
                last = last[:-1]
            else:
                lines[-1] = "..."

    return lines


def draw_wrapped(
    img: np.ndarray,
    text: str,
    box: tuple[int, int, int, int],
    font_scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    max_lines: int = 3,
    line_spacing: float = 1.2,
    align: str = "left",
    valign: str = "top",
) -> int:
    """Draw wrapped text in box. Returns number of lines drawn."""
    lines = wrap_text(text, box, font_scale, thickness, max_lines)
    if not lines:
        return 0

    x, y, w, h = box
    line_h = int(text_size("Ag", font_scale, thickness)[1] * line_spacing)
    total_h = line_h * len(lines)

    # Vertical start
    if valign == "middle":
        start_y = (h - total_h) // 2
    elif valign == "bottom":
        start_y = h - total_h
    else:
        start_y = 0

    for i, line in enumerate(lines):
        line_y = start_y + i * line_h + line_h
        # Clip line_y to box
        if line_y > h - 2:
            break
        tx = 2
        if align == "center":
            tw, _ = text_size(line, font_scale, thickness)
            tx = (w - tw) // 2
        elif align == "right":
            tw, _ = text_size(line, font_scale, thickness)
            tx = w - tw - 2

        cv2.putText(img, line, (tx, line_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, LINE_AA)

    return len(lines)


if __name__ == "__main__":
    import numpy as np
    import cv2

    print("Testing textfit...")

    # Test 1: Long string fits
    img = np.full((100, 400, 3), 30, dtype=np.uint8)
    fit_text(img, "Short text", (0, 0, 400, 100), 0.8, 0.2, (255, 255, 255))
    assert img[10, 10, 0] > 30  # Something drawn

    # Test 2: Very long string truncated
    img2 = np.full((100, 200, 3), 30, dtype=np.uint8)
    fit_text(img2, "A" * 300, (0, 0, 200, 100), 0.8, 0.2, (255, 255, 255))
    # Check ellipsis drawn
    assert img2[10, 190, 0] > 30

    # Test 3: Long unbroken word
    img3 = np.full((100, 200, 3), 30, dtype=np.uint8)
    fit_text(img3, "Supercalifragilisticexpialidocious", (0, 0, 200, 100), 0.8, 0.2, (255, 255, 255))

    # Test 4: Numbers like "-123.456 dB"
    img4 = np.full((50, 200, 3), 30, dtype=np.uint8)
    fit_text(img4, "-123.456 dB", (0, 0, 200, 50), 0.6, 0.2, (0, 255, 0))

    # Test 5: Empty string
    img5 = np.full((50, 200, 3), 30, dtype=np.uint8)
    scale = fit_text(img5, "", (0, 0, 200, 50), 0.6, 0.2, (255, 255, 255))
    assert scale == 0.6

    # Test 6: wrap_text
    lines = wrap_text("This is a long message that should wrap across multiple lines", (0, 0, 200, 100), 0.5, max_lines=3)
    assert len(lines) <= 3
    print(f"Wrapped lines: {lines}")

    # Test 7: draw_wrapped
    img6 = np.full((150, 300, 3), 30, dtype=np.uint8)
    n = draw_wrapped(img6, "Line one\nLine two\nLine three\nLine four", (10, 10, 280, 130), 0.5, (255, 255, 255), max_lines=3)
    assert n == 3

    # Test 8: Sentinel test - pixels outside box unchanged
    sentinel = 123
    img7 = np.full((100, 200, 3), sentinel, dtype=np.uint8)
    box_img = img7[10:90, 10:190]  # sub-image view
    fit_text(box_img, "Test", (0, 0, 180, 80), 0.6, 0.2, (255, 255, 255))
    # Check outside box unchanged
    assert img7[5, 5, 0] == sentinel
    assert img7[95, 195, 0] == sentinel

    print("All textfit tests passed!")