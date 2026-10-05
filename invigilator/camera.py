"""Camera source abstraction: auto-discovery, video file, and fake source for tests."""

from typing import Protocol, Optional, Any
import cv2


class FrameSource(Protocol):
    """Protocol for frame sources (camera, video file, fake)."""

    def read(self) -> tuple[bool, Optional[Any]]:
        """Read next frame. Returns (success, frame) or (False, None) if ended."""
        ...

    def release(self) -> None:
        """Release resources."""
        ...


class CameraSource:
    """OpenCV camera wrapper implementing FrameSource."""

    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera index {index} failed to open")

    def read(self) -> tuple[bool, Optional[Any]]:
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


class VideoFileSource:
    """Video file source for offline replays."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video file '{path}' failed to open")

    def read(self) -> tuple[bool, Optional[Any]]:
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


class FakeSource:
    """Fake source for testing - yields frames from a list."""

    def __init__(self, frames: list):
        self.frames = frames
        self.idx = 0

    def read(self) -> tuple[bool, Optional[Any]]:
        if self.idx < len(self.frames):
            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame
        return False, None

    def release(self) -> None:
        pass


def open_camera(preferred: int = 0, scan: range = range(0, 4)) -> FrameSource:
    """Try preferred camera index first, then auto-scan others.

    Args:
        preferred: Preferred camera index to try first
        scan: Range of indices to scan if preferred fails

    Returns:
        FrameSource instance

    Raises:
        SystemExit: If no working camera found
    """
    order = [preferred] + [i for i in scan if i != preferred]
    for idx in order:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"Using camera index {idx}")
                return CameraSource(idx)
        cap.release()

    raise SystemExit(
        f"No camera found at indices {list(scan)}. Check that it is not in use "
        "by another app and that your OS allows the terminal/IDE to access the camera "
        "(see README: Troubleshooting)."
    )


def open_source(source: Optional[str], preferred_cam: int = 0,
                scan: range = range(0, 4)) -> FrameSource:
    """Open a frame source from string spec.

    Args:
        source: None/camera index (int) or video file path (str)
        preferred_cam: Preferred camera index if source is None
        scan: Camera scan range

    Returns:
        FrameSource instance
    """
    if source is None:
        return open_camera(preferred_cam, scan)

    # Try as integer (camera index)
    try:
        idx = int(source)
        return open_camera(idx, scan)
    except ValueError:
        # Treat as file path
        return VideoFileSource(source)


if __name__ == "__main__":
    # Quick smoke test with fake source
    import numpy as np

    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.ones((100, 100, 3), dtype=np.uint8) * 255,
    ]
    src = FakeSource(frames)

    ok, frame = src.read()
    print(f"Frame 1: ok={ok}, shape={frame.shape if frame is not None else None}")
    ok, frame = src.read()
    print(f"Frame 2: ok={ok}, shape={frame.shape if frame is not None else None}")
    ok, frame = src.read()
    print(f"Frame 3: ok={ok}, frame={frame}")
    src.release()