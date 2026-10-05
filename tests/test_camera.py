"""Tests for camera.py - camera auto-discovery and frame sources."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from invigilator.camera import (
    CameraSource,
    VideoFileSource,
    FakeSource,
    open_camera,
    open_source,
    FrameSource,
)


def test_fake_source_yields_frames():
    """FakeSource yields frames then ends."""
    frames = [
        np.zeros((50, 50, 3), dtype=np.uint8),
        np.ones((50, 50, 3), dtype=np.uint8) * 128,
        np.ones((50, 50, 3), dtype=np.uint8) * 255,
    ]
    src = FakeSource(frames)

    ok1, f1 = src.read()
    assert ok1
    assert f1.shape == (50, 50, 3)
    assert np.all(f1 == 0)

    ok2, f2 = src.read()
    assert ok2
    assert np.all(f2 == 128)

    ok3, f3 = src.read()
    assert ok3
    assert np.all(f3 == 255)

    ok4, f4 = src.read()
    assert not ok4
    assert f4 is None

    src.release()  # Should not raise


def test_fake_source_empty():
    """FakeSource with empty list returns False immediately."""
    src = FakeSource([])
    ok, frame = src.read()
    assert not ok
    assert frame is None


def test_camera_source_reads():
    """CameraSource can read from camera (mocked)."""
    with patch('cv2.VideoCapture') as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap_class.return_value = mock_cap

        src = CameraSource(0)
        ok, frame = src.read()
        assert ok
        assert frame is not None

        src.release()
        mock_cap.release.assert_called_once()


def test_camera_source_fails_on_bad_index():
    """CameraSource raises on invalid index."""
    with patch('cv2.VideoCapture') as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap

        with pytest.raises(RuntimeError, match="failed to open"):
            CameraSource(999)


def test_video_file_source_reads():
    """VideoFileSource reads from video file (mocked)."""
    with patch('cv2.VideoCapture') as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_cap_class.return_value = mock_cap

        src = VideoFileSource("test.mp4")
        ok, frame = src.read()
        assert ok
        assert frame is not None

        ok, frame = src.read()
        assert not ok

        src.release()


def test_video_file_source_fails_on_bad_path():
    """VideoFileSource raises on invalid path."""
    with patch('cv2.VideoCapture') as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap

        with pytest.raises(RuntimeError, match="failed to open"):
            VideoFileSource("nonexistent.mp4")


def test_open_camera_tries_preferred_first():
    """open_camera tries preferred index first, then others."""
    with patch('cv2.VideoCapture') as mock_cap_class:
        # Create two mock capture objects
        mock_cap0 = MagicMock()
        mock_cap0.isOpened.return_value = True
        mock_cap0.read.return_value = (False, None)

        mock_cap1 = MagicMock()
        mock_cap1.isOpened.return_value = True
        mock_cap1.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

        # The first call to VideoCapture (for index 0) returns mock_cap0
        # The second call (for index 1) returns mock_cap1
        # The third call (when CameraSource is created) needs another mock
        mock_cap2 = MagicMock()
        mock_cap2.isOpened.return_value = True
        mock_cap2.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

        mock_cap_class.side_effect = [mock_cap0, mock_cap1, mock_cap2]

        src = open_camera(preferred=0, scan=range(0, 3))
        assert isinstance(src, CameraSource)

        # Should have tried index 0 first (in open_camera), then 1 (in open_camera), then 1 again (in CameraSource)
        calls = mock_cap_class.call_args_list
        assert calls[0][0][0] == 0  # First call with index 0 (in open_camera)
        assert calls[1][0][0] == 1  # Second call with index 1 (in open_camera)
        assert calls[2][0][0] == 1  # Third call with index 1 (in CameraSource)


def test_open_camera_all_fail():
    """open_camera raises SystemExit when all cameras fail."""
    with patch('cv2.VideoCapture') as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap

        with pytest.raises(SystemExit) as exc:
            open_camera(preferred=0, scan=range(0, 2))
        assert "No camera found" in str(exc.value)


def test_open_source_none_uses_camera():
    """open_source with None uses camera auto-discovery."""
    with patch('invigilator.camera.open_camera') as mock_open:
        mock_open.return_value = MagicMock(spec=FrameSource)
        src = open_source(None, preferred_cam=1)
        mock_open.assert_called_once_with(1, range(0, 4))


def test_open_source_int_string_uses_camera():
    """open_source with integer string uses camera index."""
    with patch('invigilator.camera.open_camera') as mock_open:
        mock_open.return_value = MagicMock(spec=FrameSource)
        src = open_source("2", preferred_cam=0)
        mock_open.assert_called_once_with(2, range(0, 4))


def test_open_source_string_path_uses_video():
    """open_source with non-integer string uses video file."""
    with patch('invigilator.camera.VideoFileSource') as mock_vfs:
        mock_vfs.return_value = MagicMock(spec=FrameSource)
        src = open_source("video.mp4", preferred_cam=0)
        mock_vfs.assert_called_once_with("video.mp4")


def test_open_source_negative_int_string():
    """open_source with negative int string treats as camera index."""
    with patch('invigilator.camera.open_camera') as mock_open:
        mock_open.return_value = MagicMock(spec=FrameSource)
        src = open_source("-1", preferred_cam=0)
        mock_open.assert_called_once_with(-1, range(0, 4))


def test_frame_source_protocol():
    """All sources implement FrameSource protocol."""
    # FakeSource
    fake = FakeSource([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert hasattr(fake, 'read')
    assert hasattr(fake, 'release')
    fake.release()

    # CameraSource (mocked)
    with patch('cv2.VideoCapture') as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        mock_cap_class.return_value = mock_cap
        cam = CameraSource(0)
        assert hasattr(cam, 'read')
        assert hasattr(cam, 'release')
        cam.release()

    # VideoFileSource (mocked)
    with patch('cv2.VideoCapture') as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        mock_cap_class.return_value = mock_cap
        vid = VideoFileSource("test.mp4")
        assert hasattr(vid, 'read')
        assert hasattr(vid, 'release')
        vid.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])