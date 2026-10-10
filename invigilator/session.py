"""Temporal session logic: hold timers, cooldowns, out-of-frame detection, flag events.

Pure logic - time passed in as `now`, no I/O, no camera, no file access.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List
import cv2
import os
from datetime import datetime

from invigilator.baseline import Baseline
from invigilator.config import (
    GESTURES,
    COOLDOWN_S,
    ABSENT_GRACE_S,
    VIS_MIN,
    NOSE, L_SH, R_SH,
)
from invigilator.classifier import classify
from invigilator.features import measures
from invigilator.evidence import EvidenceLog


@dataclass(frozen=True)
class FlagEvent:
    """A gesture that has been flagged after meeting hold time."""
    gesture_key: str
    label: str
    held_seconds: float


class InvigilatorSession:
    """Manages gesture hold timers, cooldowns, and flagging with evidence logging.

    Behavior:
    - A gesture flags only after it has persisted for its hold time
    - It resets if it disappears for even one frame
    - After flagging, the same gesture cannot flag again within COOLDOWN_S
    - Landmarks None for > ABSENT_GRACE_S makes out_of_frame active (hold 3.0s)
    - Low-visibility head/shoulders treated as "no reliable pose" (no classify)
    """

    def __init__(self, out_dir: str = "evidence", enable_snapshots: bool = True):
        self.evidence = EvidenceLog(out_dir) if enable_snapshots else None
        self.baseline = Baseline()
        self.active: set[str] = set()
        self.since: dict[str, float] = {}      # gesture -> time it became active
        self.last_flag: defaultdict[str, float] = defaultdict(float)
        self.banners: list[tuple[float, str]] = []  # (expire_time, text)
        self._absent_since: Optional[float] = None
        self._last_t: Optional[float] = None
        self._enable_snapshots = enable_snapshots

    @property
    def base(self) -> Optional[dict[str, float]]:
        """Compatibility property for classifier."""
        return self.baseline.base

    @property
    def calibrated(self) -> bool:
        return self.baseline.is_calibrated

    def calibrate(self, lm: list) -> dict[str, float]:
        """Capture current posture as baseline."""
        return self.baseline.calibrate(lm)

    def update(self, lm: Optional[list], frame: Optional[object] = None, now: float = 0.0) -> list:
        """Process one frame. Returns list of newly flagged events.

        Args:
            lm: pose landmarks or None
            frame: video frame (for snapshots), optional
            now: current timestamp (required for temporal logic)

        For backward compatibility, can be called as update(lm, now) or update(lm, frame, now).
        """
        # Handle backward compatibility: update(lm, now) -> frame=None, now=frame
        if frame is not None and now == 0.0 and not isinstance(frame, (int, float)):
            # Called as update(lm, frame, now) - correct order
            pass
        elif frame is not None and isinstance(frame, (int, float)):
            # Called as update(lm, now) - frame is actually now
            now = frame
            frame = None
        elif frame is None and now == 0.0:
            raise TypeError("update() missing required argument 'now'")

        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        active: set[str] = set()

        if lm is not None:
            # Check if key landmarks are visible enough
            if min(lm[i].visibility for i in (NOSE, L_SH, R_SH)) > VIS_MIN:
                active = classify(lm, self.baseline.base)
                # Adaptive baseline: only when no gesture active AND near neutral
                if self.baseline.base is not None and not active:
                    m = measures(lm)
                    self.baseline.adapt(m, dt)
            # Person visible but key landmarks not reliable -> no classify, no adapt
            self._absent_since = None
        else:
            # No person detected
            self._absent_since = self._absent_since or now
            if now - self._absent_since > ABSENT_GRACE_S:
                active.add("out_of_frame")

        # Update active set and since timers
        self.active = active
        for g in list(self.since):
            if g not in active:
                del self.since[g]

        # Check for flags
        new_flags = []
        for g in active:
            self.since.setdefault(g, now)
            cfg = GESTURES[g]
            label, need = cfg.label, cfg.hold_seconds
            held = now - self.since[g]
            if held >= need and now - self.last_flag[g] >= COOLDOWN_S:
                self.last_flag[g] = now
                flag_event = FlagEvent(g, label, held)
                new_flags.append(flag_event)
                self.banners.append((now + 4.0, f"FLAG: {label}"))
                # Record to evidence log
                if self.evidence and frame is not None:
                    self._record(g, label, held, frame)

        # Prune expired banners
        self.banners = [(t, s) for t, s in self.banners if t > now]

        return new_flags

    def _record(self, gesture: str, label: str, held: float, frame: object) -> None:
        """Record a flagged event to evidence log."""
        if not self._enable_snapshots or not self.evidence:
            return
        self.evidence.record(gesture, label, held, frame)

    def _record_audio_alert(self, alert, frame: object) -> None:
        """Record an audio alert to evidence log."""
        if not self._enable_snapshots or not self.evidence:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = os.path.join(self.evidence.out_dir, f"{ts}_audio_{alert.audio_kind}.jpg")
        cv2.imwrite(snap, frame)
        # Append to av_events.csv
        av_log_path = os.path.join(self.evidence.out_dir, "av_events.csv")
        is_new = not os.path.exists(av_log_path)
        with open(av_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "severity", "risk", "audio_kind", "video_cues", "reasons", "snapshot"])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                alert.severity,
                f"{alert.risk:.2f}",
                alert.audio_kind,
                "|".join(alert.video_cues),
                "|".join(alert.reasons),
                snap
            ])

    def get_view(self) -> dict:
        """Return read-only snapshot for overlay rendering."""
        return {
            "active": frozenset(self.active),
            "since": dict(self.since),
            "banners": list(self.banners),
            "calibrated": self.baseline.is_calibrated,
        }

    def close(self) -> None:
        """Close evidence log."""
        if self.evidence:
            self.evidence.close()
            self.evidence = None


# Backward compatibility aliases
Session = InvigilatorSession
FlagEvent = FlagEvent


if __name__ == "__main__":
    # Quick smoke test with FakeClock
    from tests.conftest import make_landmarks, FakeClock

    clock = FakeClock(1000.0)
    session = InvigilatorSession(out_dir="test_evidence", enable_snapshots=False)
    lm = make_landmarks()

    # Neutral - no flags
    flags = session.update(lm, None, clock.now)
    print(f"t={clock.now}: neutral, flags={flags}, active={session.active}")

    # Look right for 3 seconds (threshold 2.5s)
    lm_right = make_landmarks(overrides={0: type('L', (), {'x': 0.62, 'y': 0.30, 'visibility': 1.0})()})
    clock.advance(1.0)
    flags = session.update(lm_right, None, clock.now)
    print(f"t={clock.now}: look_right 1s, flags={flags}")

    clock.advance(2.0)
    flags = session.update(lm_right, None, clock.now)
    print(f"t={clock.now}: look_right 3s, flags={flags}, held={flags[0][2] if flags else 'N/A'}")

    # Cooldown test
    clock.advance(5.5)
    flags = session.update(lm_right, None, clock.now)
    print(f"t={clock.now}: after cooldown, flags={flags}")

    session.close()