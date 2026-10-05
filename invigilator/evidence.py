"""Evidence logger: writes one CSV row and one snapshot per flag. Nothing else."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EvidenceLog:
    """Logs flagged events to CSV and saves one snapshot JPEG per flag."""

    def __init__(self, out_dir: str = "evidence"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "events.csv"
        self._log_file = open(self.log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._log_file)

        # Write header if file is new/empty
        if self.log_path.stat().st_size == 0:
            self._writer.writerow(
                ["timestamp", "gesture", "label", "held_seconds", "snapshot"]
            )
            self._log_file.flush()

    def record(
        self,
        gesture: str,
        label: str,
        held_seconds: float,
        frame: np.ndarray,
    ) -> str:
        """Write one CSV row and one JPEG snapshot for a flagged event.

        Returns the snapshot path.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_name = f"{ts}_{gesture}.jpg"
        snap_path = self.out_dir / snap_name

        # Write exactly one snapshot
        cv2.imwrite(str(snap_path), frame)

        # Write CSV row
        self._writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                gesture,
                label,
                f"{held_seconds:.1f}",
                snap_name,
            ]
        )
        self._log_file.flush()

        return str(snap_path)

    def close(self) -> None:
        """Close the log file."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Quick smoke test
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()
    try:
        log = EvidenceLog(tmpdir)
        # Create a dummy frame (black 100x100)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        snap = log.record("hand_raised", "Hand raised", 1.5, frame)
        print(f"Snapshot: {snap}")
        print(f"Exists: {os.path.exists(snap)}")

        # Check CSV
        with open(Path(tmpdir) / "events.csv") as f:
            print("CSV content:", f.read())
        log.close()
    finally:
        shutil.rmtree(tmpdir)