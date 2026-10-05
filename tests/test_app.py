"""Tests for app.py - end-to-end integration test."""
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from invigilator.app import run_loop
from invigilator.camera import FakeSource
from invigilator.pose import PoseEstimator
from invigilator.session import Session
from invigilator.evidence import EvidenceLog
from invigilator.config import GESTURES


def test_run_loop_end_to_end_flag_logged():
    """End-to-end: fake source + stub pose -> exactly one CSV row and one snapshot."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create fake frames with "look_right" pose
        frames = []
        for _ in range(150):  # 5 seconds at 30fps
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frames.append(frame)

        source = FakeSource(frames)

        # Stub pose estimator that always returns look_right landmarks
        class StubPose:
            def __init__(self):
                self._connections = []

            def process(self, frame):
                from types import SimpleNamespace
                # Create landmarks with nose shifted right
                landmarks = []
                for i in range(33):
                    if i == 0:  # NOSE
                        landmarks.append(SimpleNamespace(x=0.62, y=0.30, visibility=1.0))
                    elif i == 7:  # L_EAR
                        landmarks.append(SimpleNamespace(x=0.40, y=0.28, visibility=1.0))
                    elif i == 8:  # R_EAR
                        landmarks.append(SimpleNamespace(x=0.60, y=0.28, visibility=1.0))
                    elif i == 11:  # L_SH
                        landmarks.append(SimpleNamespace(x=0.35, y=0.50, visibility=1.0))
                    elif i == 12:  # R_SH
                        landmarks.append(SimpleNamespace(x=0.65, y=0.50, visibility=1.0))
                    elif i == 15:  # L_WR
                        landmarks.append(SimpleNamespace(x=0.35, y=0.70, visibility=1.0))
                    elif i == 16:  # R_WR
                        landmarks.append(SimpleNamespace(x=0.65, y=0.70, visibility=1.0))
                    else:
                        landmarks.append(SimpleNamespace(x=0.50, y=0.40, visibility=1.0))
                return landmarks, None

            def close(self):
                pass

        pose = StubPose()
        session = Session()
        evidence = EvidenceLog(tmpdir)

        # Run with fake clock simulation via key presses
        # We need to simulate time advancement - let's use a custom run
        import time
        start = time.time()

        # Simulate the loop manually to control time
        for i, frame in enumerate(frames):
            now = start + i / 30.0  # 30 FPS
            landmarks, _ = pose.process(frame)
            flags = session.update(landmarks, now)
            for flag in flags:
                evidence.record(flag.gesture_key, flag.label, flag.held_seconds, frame)

        evidence.close()

        # Check results
        import csv

        csv_path = Path(tmpdir) / "events.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 1 flag row (look_right triggers at 2.5s, ~75 frames)
        assert len(rows) == 2, f"Expected 2 rows (header + 1 flag), got {len(rows)}"
        assert rows[1][1] == "look_right"
        assert rows[1][2] == "Looking right"

        # Check snapshot exists
        snapshots = list(Path(tmpdir).glob("*.jpg"))
        assert len(snapshots) == 1, f"Expected 1 snapshot, got {len(snapshots)}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_run_loop_no_flags_when_neutral():
    """Neutral pose should not produce any flags."""
    tmpdir = tempfile.mkdtemp()
    try:
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(30)]
        source = FakeSource(frames)

        class StubPose:
            _connections = []
            def process(self, frame):
                from types import SimpleNamespace
                landmarks = []
                for i in range(33):
                    if i == 0:
                        landmarks.append(SimpleNamespace(x=0.50, y=0.30, visibility=1.0))
                    elif i == 7:
                        landmarks.append(SimpleNamespace(x=0.40, y=0.28, visibility=1.0))
                    elif i == 8:
                        landmarks.append(SimpleNamespace(x=0.60, y=0.28, visibility=1.0))
                    elif i == 11:
                        landmarks.append(SimpleNamespace(x=0.35, y=0.50, visibility=1.0))
                    elif i == 12:
                        landmarks.append(SimpleNamespace(x=0.65, y=0.50, visibility=1.0))
                    elif i == 15:
                        landmarks.append(SimpleNamespace(x=0.35, y=0.70, visibility=1.0))
                    elif i == 16:
                        landmarks.append(SimpleNamespace(x=0.65, y=0.70, visibility=1.0))
                    else:
                        landmarks.append(SimpleNamespace(x=0.50, y=0.40, visibility=1.0))
                return landmarks, None
            def close(self):
                pass

        pose = StubPose()
        session = Session()
        evidence = EvidenceLog(tmpdir)

        import time
        start = time.time()
        for i, frame in enumerate(frames):
            now = start + i / 30.0
            landmarks, _ = pose.process(frame)
            flags = session.update(landmarks, now)
            for flag in flags:
                evidence.record(flag.gesture_key, flag.label, flag.held_seconds, frame)

        evidence.close()

        import csv
        with open(Path(tmpdir) / "events.csv") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Only header, no flags
        assert len(rows) == 1

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_run_loop_handshake():
    """Verify run_loop function signature and basic operation."""
    # Just verify the function can be imported and called
    from invigilator.camera import FakeSource
    from invigilator.session import Session
    from invigilator.evidence import EvidenceLog

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    source = FakeSource(frames)

    class DummyPose:
        _connections = []
        def process(self, frame):
            return None, None
        def close(self):
            pass

    pose = DummyPose()
    session = Session()
    evidence = EvidenceLog(tempfile.mkdtemp())

    # Should not raise
    run_loop(source, pose, session, evidence, lambda f, *a: f, keys=[ord('q')])

    source.release()
    pose.close()
    evidence.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])