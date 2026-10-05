"""MediaPipe Pose wrapper: isolates MediaPipe behind one class."""

from typing import Optional, Tuple, Any
import cv2
import mediapipe as mp


class PoseEstimator:
    """Wraps MediaPipe Pose for landmark extraction and drawing."""

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            self._pose = mp.solutions.pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except AttributeError as e:
            raise RuntimeError(
                "MediaPipe solutions not found. "
                "Ensure mediapipe==0.10.14 is installed (see requirements.txt)."
            ) from e

        self._drawer = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._connections = mp.solutions.pose.POSE_CONNECTIONS

    def process(self, bgr_frame: Any) -> Tuple[Optional[list], Any]:
        """Process a BGR frame, return (landmarks_list, raw_result).

        Args:
            bgr_frame: OpenCV BGR image (numpy array)

        Returns:
            Tuple of (landmarks_list_or_None, raw_mediapipe_result)
            landmarks_list: list of 33 SimpleNamespace(x, y, visibility) or None
            raw_result: MediaPipe pose result (for drawing)
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            return list(landmarks), result
        return None, result

    def draw_landmarks(self, frame: Any, result: Any) -> None:
        """Draw pose landmarks on frame in-place."""
        if result and result.pose_landmarks:
            self._drawer.draw_landmarks(
                frame,
                result.pose_landmarks,
                self._connections,
                landmark_drawing_spec=self._styles.get_default_pose_landmarks_style(),
            )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()
            self._pose = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np

    estimator = PoseEstimator()

    # Black frame - should return None landmarks
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    landmarks, result = estimator.process(black)
    print(f"Black frame: landmarks={landmarks}, has_result={result is not None}")

    estimator.close()
    print("PoseEstimator closed successfully")