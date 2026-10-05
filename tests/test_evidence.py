"""Tests for evidence.py - CSV logging and snapshot saving."""
import csv
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from invigilator.evidence import EvidenceLog


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def dummy_frame():
    """Create a small dummy frame for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_creates_folder_and_csv(temp_dir):
    """EvidenceLog creates the output folder and events.csv."""
    log = EvidenceLog(temp_dir)
    log.close()

    assert Path(temp_dir).exists()
    assert (Path(temp_dir) / "events.csv").exists()


def test_writes_header_once(temp_dir):
    """Header written once even across re-opens."""
    log1 = EvidenceLog(temp_dir)
    log1.close()

    log2 = EvidenceLog(temp_dir)
    log2.close()

    with open(Path(temp_dir) / "events.csv") as f:
        lines = f.readlines()

    # Should have exactly one header line
    assert len(lines) == 1
    assert lines[0].strip() == "timestamp,gesture,label,held_seconds,snapshot"


def test_record_writes_row_and_snapshot(temp_dir, dummy_frame):
    """record() writes one CSV row and one JPEG snapshot."""
    log = EvidenceLog(temp_dir)
    snap_path = log.record("hand_raised", "Hand raised", 1.5, dummy_frame)
    log.close()

    # Check snapshot exists
    assert os.path.exists(snap_path)
    assert snap_path.endswith(".jpg")

    # Check CSV row
    with open(Path(temp_dir) / "events.csv") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 2  # header + 1 data row
    assert rows[1][1] == "hand_raised"
    assert rows[1][2] == "Hand raised"
    assert rows[1][3] == "1.5"
    assert rows[1][4] == os.path.basename(snap_path)


def test_multiple_records_append_rows(temp_dir, dummy_frame):
    """Multiple record() calls append rows to CSV."""
    log = EvidenceLog(temp_dir)
    log.record("hand_raised", "Hand raised", 1.5, dummy_frame)
    log.record("look_right", "Looking right", 2.6, dummy_frame)
    log.record("hand_to_face", "Hand covering face/mouth", 2.1, dummy_frame)
    log.close()

    with open(Path(temp_dir) / "events.csv") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 4  # header + 3 data rows


def test_snapshot_one_per_call(temp_dir, dummy_frame):
    """Each record() call creates exactly one snapshot file."""
    log = EvidenceLog(temp_dir)

    snap1 = log.record("hand_raised", "Hand raised", 1.5, dummy_frame)
    snap2 = log.record("look_right", "Looking right", 2.6, dummy_frame)

    assert os.path.exists(snap1)
    assert os.path.exists(snap2)
    assert snap1 != snap2  # Different timestamps

    # Count jpg files
    jpgs = list(Path(temp_dir).glob("*.jpg"))
    assert len(jpgs) == 2

    log.close()


def test_unwritable_folder_raises_error(temp_dir):
    """Unwritable folder raises a clear error."""
    # On Windows, os.chmod doesn't work the same way. Skip this test on Windows.
    import sys
    if sys.platform == "win32":
        pytest.skip("chmod not reliable on Windows")

    # Make directory read-only
    os.chmod(temp_dir, 0o555)

    try:
        log = EvidenceLog(temp_dir)
        # On Windows, this might not fail immediately, but writing will
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises((PermissionError, OSError)):
            log.record("hand_raised", "Hand raised", 1.5, frame)
    finally:
        os.chmod(temp_dir, 0o777)


def test_no_extra_files_created(temp_dir, dummy_frame):
    """Privacy test: only events.csv and .jpg snapshots are created."""
    log = EvidenceLog(temp_dir)
    log.record("hand_raised", "Hand raised", 1.5, dummy_frame)
    log.record("look_right", "Looking right", 2.6, dummy_frame)
    log.close()

    all_files = list(Path(temp_dir).rglob("*"))
    # Should have: events.csv + 2 jpgs = 3 files
    assert len(all_files) == 3

    for f in all_files:
        assert f.suffix in (".csv", ".jpg"), f"Unexpected file type: {f}"


def test_context_manager(temp_dir, dummy_frame):
    """EvidenceLog works as context manager."""
    with EvidenceLog(temp_dir) as log:
        log.record("hand_raised", "Hand raised", 1.5, dummy_frame)

    # Should be closed automatically
    with open(Path(temp_dir) / "events.csv") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2


def test_close_idempotent(temp_dir):
    """close() can be called multiple times without error."""
    log = EvidenceLog(temp_dir)
    log.close()
    log.close()  # Should not raise


def test_snapshot_filename_format(temp_dir, dummy_frame):
    """Snapshot filename follows YYYYMMDD_HHMMSS_gesture.jpg format."""
    log = EvidenceLog(temp_dir)
    snap = log.record("hand_raised", "Hand raised", 1.5, dummy_frame)
    log.close()

    filename = os.path.basename(snap)
    # Format: YYYYMMDD_HHMMSS_gesture.jpg (gesture may contain underscores)
    assert filename.endswith("_hand_raised.jpg")
    assert len(filename) >= 8 + 1 + 6 + 1 + len("hand_raised.jpg")
    # Verify date and time parts
    date_part = filename[:8]
    time_part = filename[9:15]
    assert date_part.isdigit() and len(date_part) == 8
    assert time_part.isdigit() and len(time_part) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])