"""Findings recorder: accumulates session data and exports summary report.

Pure functions: time injected, no I/O in core logic.
"""
import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

from ui.panels import PanelState, GestureProgress


# =============================================================================
# Data classes for event records
# =============================================================================

@dataclass(frozen=True)
class GestureEvent:
    """A single gesture activation period."""
    key: str
    label: str
    start_time: float
    end_time: float
    was_flagged: bool  # held long enough to flag


@dataclass(frozen=True)
class AudioEvent:
    """A single audio event."""
    kind: str  # "whispering" / "talking"
    start_time: float
    end_time: float
    mean_confidence: float
    coverage_seconds: float


@dataclass(frozen=True)
class FusedAlert:
    """A fused alert."""
    time: float
    severity: str  # "alert" / "review"
    risk: float
    audio_kind: str
    video_cues: List[str]
    reasons: List[str]


@dataclass(frozen=True)
class FrameStateEvent:
    """Frame state change."""
    time: float
    state: str  # "IN" / "PARTIAL" / "OUT"


# =============================================================================
# FindingsRecorder
# =============================================================================

class FindingsRecorder:
    """Accumulates session data for export.

    Pure logic: time passed in as `now`, no I/O in update().
    Export methods write files.
    """

    def __init__(self, session_start: float, consent_path: Optional[str] = None):
        self.session_start = session_start
        self.consent_path = consent_path
        self.session_end: Optional[float] = None

        # Gesture tracking
        self._gesture_active: Dict[str, float] = {}  # key -> start_time
        self._gesture_events: List[GestureEvent] = []
        self._gesture_last_flag_time: Dict[str, float] = {}  # for cooldown tracking

        # Frame state tracking
        self._frame_state: str = "OUT"
        self._frame_state_events: List[FrameStateEvent] = []

        # Audio tracking
        self._audio_events: List = []
        self._fused_alerts: List = []

        # Movement tracking (body/hand)
        self._movement_time: Dict[str, float] = {"STILL": 0.0, "LOW": 0.0, "HIGH": 0.0}
        self._hand_movement_time: Dict[str, float] = {"STILL": 0.0, "MOVING": 0.0, "FAST": 0.0}
        self._last_movement: Optional[str] = None
        self._last_movement_time: Optional[float] = None
        self._last_hand_movement: Optional[str] = None
        self._last_hand_movement_time: Optional[float] = None

        # Audio label tracking
        self._audio_label_time: Dict[str, float] = {}
        self._last_audio_label: Optional[str] = None
        self._last_audio_label_time: Optional[float] = None

        # Calibration
        self._calibration_status: str = "unknown"
        self._peak_level_db: float = -100.0

        # Flags count
        self._flags_count: int = 0

    # -------------------------------------------------------------------------
    # Update methods (called each frame with injected time)
    # -------------------------------------------------------------------------

    def update_video(self, panel_state: PanelState, now: float) -> None:
        """Update from video panel state."""
        # Track gesture progress
        active_keys = {gp.key for gp in panel_state.gesture_progress}

        # New gestures started
        for key in active_keys:
            if key not in self._gesture_active:
                self._gesture_active[key] = now

        # Gestures ended
        for key in list(self._gesture_active.keys()):
            if key not in active_keys:
                start = self._gesture_active.pop(key)
                # Find if it was flagged
                was_flagged = any(e.key == key and e.was_flagged for e in self._gesture_events)
                self._gesture_events.append(GestureEvent(
                    key=key,
                    label=next((gp.label for gp in panel_state.gesture_progress if gp.key == key), key),
                    start_time=start,
                    end_time=now,
                    was_flagged=was_flagged,
                ))

        # Track flags count from panel
        self._flags_count = panel_state.flags_count

        # Frame state changes
        if panel_state.frame_state != self._frame_state:
            self._frame_state_events.append(FrameStateEvent(
                time=now, state=panel_state.frame_state
            ))
            self._frame_state = panel_state.frame_state

        # Movement time shares - track precisely
        self._last_movement, self._last_movement_time = self._accumulate_time_precise(
            panel_state.body_movement, self._movement_time,
            self._last_movement, self._last_movement_time, now
        )
        self._last_hand_movement, self._last_hand_movement_time = self._accumulate_time_precise(
            panel_state.hand_movement, self._hand_movement_time,
            self._last_hand_movement, self._last_hand_movement_time, now
        )

    def update_audio(self, panel_state: PanelState, now: float) -> None:
        """Update from audio panel state."""
        # Audio label time shares - track precisely
        self._last_audio_label, self._last_audio_label_time = self._accumulate_time_precise(
            panel_state.audio_label, self._audio_label_time,
            self._last_audio_label, self._last_audio_label_time, now
        )

        # Calibration status
        self._calibration_status = panel_state.calibration_status
        self._peak_level_db = max(self._peak_level_db, panel_state.level_db_above_floor)

    def update_fused_alert(self, alert, now: float) -> None:
        """Record a fused alert."""
        self._fused_alerts.append(alert)

    def record_audio_event(self, event) -> None:
        """Record an audio event."""
        self._audio_events.append(event)

    def _accumulate_time_precise(self, current: str, accum: Dict[str, float],
                                  last_key: Optional[str], last_time: Optional[float],
                                  now: float):
        """Accumulate time and return new (last_key, last_time)."""
        # Ensure key exists in accumulator
        if current not in accum:
            accum[current] = 0.0
        if last_key is not None and last_time is not None and last_key in accum:
            accum[last_key] += now - last_time
        return current, now

    # -------------------------------------------------------------------------
    # Export methods
    # -------------------------------------------------------------------------

    def finalize(self, now: float) -> None:
        """Call at session end to finalize timing."""
        self.session_end = now

        # Close any open gesture intervals
        for key, start in self._gesture_active.items():
            self._gesture_events.append(GestureEvent(
                key=key,
                label=key,
                start_time=start,
                end_time=now,
                was_flagged=False,
            ))
        self._gesture_active.clear()

        # Final frame state
        self._frame_state_events.append(FrameStateEvent(time=now, state=self._frame_state))

    def export_markdown(self, output_dir: str = "evidence") -> str:
        """Export findings to markdown file. Returns filepath."""
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.fromtimestamp(self.session_start).strftime("%Y%m%d_%H%M%S")
        md_path = os.path.join(output_dir, "findings_{}.md".format(ts))
        csv_path = os.path.join(output_dir, "findings_{}.csv".format(ts))

        duration = (self.session_end or self.session_start) - self.session_start

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown(duration, csv_path))

        # Also write CSV events table
        self._write_events_csv(csv_path)

        return md_path

    def _generate_markdown(self, duration: float, csv_path: str) -> str:
        lines = []
        lines.append("# Edge Invigilator Session Findings")
        lines.append("")
        lines.append("**Session start:** {0}".format(datetime.fromtimestamp(self.session_start).strftime('%Y-%m-%d %H:%M:%S')))
        lines.append("**Session end:** {0}".format(datetime.fromtimestamp(self.session_end or self.session_start).strftime('%Y-%m-%d %H:%M:%S')))
        lines.append("**Duration:** {0:.1f} seconds".format(duration))
        lines.append("")

        # Video summary
        lines.append("## Video Detection Summary")
        lines.append("")

        # Frame state time shares
        total_frame = 0.0
        if len(self._frame_state_events) > 1:
            for i in range(len(self._frame_state_events)-1):
                total_frame += self._frame_state_events[i+1].time - self._frame_state_events[i].time
        else:
            total_frame = duration

        if total_frame > 0:
            for state in ["IN", "PARTIAL", "OUT"]:
                state_time = 0.0
                for i in range(len(self._frame_state_events)-1):
                    if self._frame_state_events[i].state == state:
                        state_time += self._frame_state_events[i+1].time - self._frame_state_events[i].time
                if state_time > 0:
                    lines.append("- Frame {0}: {1:.1f}s ({2:.1f}%)".format(state, state_time, 100.0*state_time/total_frame))
        lines.append("")

        # Gesture summary
        lines.append("### Gestures")
        lines.append("")
        for g in self._gesture_events:
            flag_str = " 🚩 FLAGGED" if g.was_flagged else ""
            dur = g.end_time - g.start_time
            lines.append("- {0}: {1:.1f}s total{2}".format(g.label, dur, flag_str))
        lines.append("")

        # Face direction
        face_dirs = {}
        for g in self._gesture_events:
            if g.key.startswith("look_"):
                face_dirs[g.key] = face_dirs.get(g.key, 0) + (g.end_time - g.start_time)
        if face_dirs:
            lines.append("### Face Direction")
            for k, v in face_dirs.items():
                lines.append("- {0}: {1:.1f}s".format(k.replace('look_', '').title(), v))
            lines.append("")

        # Body/hand movement
        total_mov = sum(self._movement_time.values())
        if total_mov > 0:
            lines.append("### Body Movement")
            for k, v in self._movement_time.items():
                lines.append("- {0}: {1:.1f}s ({2:.1f}%)".format(k, v, 100.0*v/total_mov))
            lines.append("")

        total_hmov = sum(self._hand_movement_time.values())
        if total_hmov > 0:
            lines.append("### Hand Movement")
            for k, v in self._hand_movement_time.items():
                lines.append("- {0}: {1:.1f}s ({2:.1f}%)".format(k, v, 100.0*v/total_hmov))
            lines.append("")

        # Audio summary
        lines.append("## Audio Detection Summary")
        lines.append("")
        lines.append("- Calibration: {0}".format(self._calibration_status))
        lines.append("- Peak level above floor: {0:.1f} dB".format(self._peak_level_db))
        lines.append("")

        # Audio label time shares
        total_audio = sum(self._audio_label_time.values())
        if total_audio > 0:
            lines.append("### Audio Labels")
            items = sorted(self._audio_label_time.items(), key=lambda x: -x[1])
            for label, t in items:
                lines.append("- {0}: {1:.1f}s ({2:.1f}%)".format(label, t, 100.0*t/total_audio))
            lines.append("")

        # Audio events
        if self._audio_events:
            lines.append("### Audio Events")
            for ae in self._audio_events:
                lines.append("- {0}: {1:.1f}s coverage, conf={2:.0%}".format(ae.kind, ae.coverage_seconds, ae.mean_confidence))
            lines.append("")

        # Fused alerts
        if self._fused_alerts:
            lines.append("### Fused Alerts")
            for fa in self._fused_alerts:
                lines.append("- t={0:.1f}s: {1} risk={2:.2f} ({3}) cues={4}".format(fa.time, fa.severity.upper(), fa.risk, fa.audio_kind, ", ".join(fa.video_cues)))
            lines.append("")

        lines.append("- Flags this session: {0}".format(self._flags_count))
        lines.append("- Audio events: {0}".format(len(self._audio_events)))
        lines.append("- Fused alerts: {0}".format(len(self._fused_alerts)))
        lines.append("")

        lines.append("---")
        lines.append("**Decision-support output for human review, not proof of misconduct.**")
        lines.append("")
        lines.append("Events table: `{0}`".format(os.path.basename(csv_path)))

        return "\n".join(lines)

    def _write_events_csv(self, csv_path: str) -> None:
        """Write chronological events table to CSV."""
        events = []

        # Gesture events
        for g in self._gesture_events:
            events.append({
                "time": g.start_time,
                "type": "gesture",
                "subtype": g.key,
                "label": g.label,
                "duration": g.end_time - g.start_time,
                "flagged": g.was_flagged,
            })

        # Frame state events
        for fs in self._frame_state_events:
            events.append({
                "time": fs.time,
                "type": "frame_state",
                "subtype": fs.state,
                "label": fs.state,
                "duration": 0,
                "flagged": False,
            })

        # Audio events
        for ae in self._audio_events:
            events.append({
                "time": ae.start_time,
                "type": "audio",
                "subtype": ae.kind,
                "label": ae.kind,
                "duration": ae.end_time - ae.start_time,
                "flagged": False,
            })

        # Fused alerts
        for fa in self._fused_alerts:
            events.append({
                "time": fa.time,
                "type": "fused_alert",
                "subtype": fa.severity,
                "label": "{0} + {2}".format(fa.audio_kind, ",".join(fa.video_cues)),
                "duration": 0,
                "flagged": fa.severity == "alert",
            })

        # Sort by time
        events.sort(key=lambda e: e["time"])

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "type", "subtype", "label", "duration", "flagged"])
            writer.writeheader()
            for e in events:
                writer.writerow(e)

    # -------------------------------------------------------------------------
    # Consent check
    # -------------------------------------------------------------------------

    def can_export(self) -> bool:
        """Check if export is allowed (consent file exists)."""
        if self.consent_path and os.path.exists(self.consent_path):
            return True
        return False


# =============================================================================
# Convenience: build recorder from session data
# =============================================================================

def create_findings_recorder(
    session_start: float,
    consent_path: Optional[str] = None,
) -> 'FindingsRecorder':
    """Create a FindingsRecorder with default settings."""
    return FindingsRecorder(session_start, consent_path)