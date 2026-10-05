"""Tests for pose.py - MediaPipe Pose wrapper."""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from invigilator.pose import PoseEstimator


def test_black_frame_returns_none_landmarks():
    """Black frame should return None landmarks."""
    estimator = PoseEstimator()
    black = np.zeros((240, 320, 3), dtype=np.uint8)

    landmarks, result = estimator.process(black)

    assert landmarks is None
    assert result is not None
    assert result.pose_landmarks is None

    estimator.close()


def test_close_is_idempotent():
    """close() can be called multiple times without error."""
    estimator = PoseEstimator()
    estimator.close()
    estimator.close()  # Should not raise


def test_context_manager():
    """PoseEstimator works as context manager."""
    with PoseEstimator() as estimator:
        black = np.zeros((240, 320, 3), dtype=np.uint8)
        landmarks, _ = estimator.process(black)
        assert landmarks is None
    # Should be closed automatically


def test_missing_solutions_raises_helpful_error():
    """Missing mp.solutions raises error pointing to requirements.txt."""
    with patch('invigilator.pose.mp') as mock_mp:
        # Simulate missing solutions attribute
        mock_mp.solutions = None

        with pytest.raises(RuntimeError) as exc:
            PoseEstimator()
        assert "mediapipe==0.10.14" in str(exc.value)
        assert "requirements.txt" in str(exc.value)


def test_process_returns_landmarks_and_result():
    """process() returns (landmarks_list, raw_result) tuple."""
    estimator = PoseEstimator()
    black = np.zeros((240, 320, 3), dtype=np.uint8)

    landmarks, result = estimator.process(black)

    assert isinstance(landmarks, (list, type(None)))
    assert result is not None

    estimator.close()


def test_draw_landmarks_noop_on_none():
    """draw_landmarks does nothing when result has no pose_landmarks."""
    estimator = PoseEstimator()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Result with no landmarks
    class MockResult:
        pose_landmarks = None

    # Should not raise
    estimator.draw_landmarks(frame, MockResult())
    estimator.close()


def test_init_parameters():
    """PoseEstimator accepts model_complexity and confidence parameters."""
    estimator = PoseEstimator(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    estimator.close()


def test_draw_landmarks_with_mock_result():
    """draw_landmarks calls MediaPipe drawer when landmarks present."""
    estimator = PoseEstimator()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Mock the drawer
    with patch.object(estimator, '_drawer') as mock_drawer:
        class MockLandmark:
            x = 0.5
            y = 0.5
            z = 0.0
            visibility = 1.0

        class MockPoseLandmarks:
            landmark = [MockLandmark() for _ in range(33)]

        class MockResult:
            pose_landmarks = MockPoseLandmarks()

        estimator.draw_landmarks(frame, MockResult())
        mock_drawer.draw_landmarks.assert_called_once()

    estimator.close()


def test_multiple_process_calls():
    """Multiple process calls work correctly."""
    estimator = PoseEstimator()
    black = np.zeros((240, 320, 3), dtype=np.uint8)

    for _ in range(5):
        landmarks, result = estimator.process(black)
        assert landmarks is None
        assert result is not None

    estimator.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])