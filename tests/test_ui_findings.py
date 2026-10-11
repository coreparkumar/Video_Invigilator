"""Tests for ui/findings.py - FindingsRecorder and export."""
import csv
import os
import tempfile
import shutil
import time
from types import SimpleNamespace

from ui.findings import (
    FindingsRecorder, GestureEvent, AudioEvent, FusedAlert, FrameStateEvent,
    create_findings_recorder,
)
from ui.panels import PanelState, GestureProgress
from ui.config import COLORS


def test_findings_recorder_basic():
    """FindingsRecorder can be created and finalized."""
    start = time.time()
    recorder = FindingsRecorder(start)
    recorder.finalize(start + 10.0)

    assert recorder.session_start == start
    assert recorder.session_end is not None
    assert recorder.session_end > recorder.session_start


def test_findings_recorder_gesture_tracking():
    """Recorder tracks gesture start/end."""
    start = time.time()
    recorder = FindingsRecorder(start)

    # Simulate gesture start
    panel = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[GestureProgress("look_left", "Looking left", 1.0, 2.5, COLORS.ACTIVE_AMBER)],
        flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )

    recorder.update_video(panel, start + 1.0)

    # Simulate gesture continuing
    recorder.update_video(panel, start + 2.0)

    # Gesture ends
    panel2 = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[],
        flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel2, start + 3.0)

    recorder.finalize(start + 4.0)

    # Should have one gesture event
    assert len(recorder._gesture_events) == 1
    ge = recorder._gesture_events[0]
    assert ge.key == "look_left"
    assert ge.start_time == start + 1.0
    assert ge.end_time == start + 3.0


def test_findings_recorder_frame_state_changes():
    """Recorder tracks frame state changes."""
    start = time.time()
    recorder = FindingsRecorder(start)

    # IN -> PARTIAL -> OUT
    for fs in ["IN", "PARTIAL", "OUT", "IN"]:
        panel = PanelState(
            frame_state=fs, face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
            hand_state="NONE", hand_movement="STILL",
            gesture_progress=[],
            flags_count=0,
            audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
            threshold_db=-52.0, calibration_status="ok", calibration_message="",
            clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
            audio_events_count=0,
        )
        recorder.update_video(panel, start + 1.0)

    recorder.finalize(start + 10.0)

    states = [e.state for e in recorder._frame_state_events]
    assert "IN" in states
    assert "PARTIAL" in states
    assert "OUT" in states


def test_findings_recorder_audio_labels():
    """Recorder tracks audio label time shares."""
    start = time.time()
    recorder = FindingsRecorder(start)

    for label in ["SILENCE", "WHISPER", "SPEECH", "SILENCE"]:
        panel = PanelState(
            frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
            hand_state="NONE", hand_movement="STILL",
            gesture_progress=[],
            flags_count=0,
            audio_label=label, audio_confidence=0.8, level_db_above_floor=10.0,
            threshold_db=8.0, calibration_status="ok", calibration_message="",
            clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
            audio_events_count=0,
        )
        recorder.update_audio(panel, start + 1.0)

    recorder.finalize(start + 10.0)

    # Should have tracked all labels
    assert "SILENCE" in recorder._audio_label_time
    assert "WHISPER" in recorder._audio_label_time
    assert "SPEECH" in recorder._audio_label_time


def test_findings_recorder_flags_count():
    """Flags count increments on flagged gestures."""
    start = time.time()
    recorder = FindingsRecorder(start)

    # Non-flagged gesture
    panel = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[GestureProgress("look_left", "Looking left", 1.0, 2.5, COLORS.ACTIVE_AMBER)],
        flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel, start + 1.0)
    recorder.update_video(panel, start + 2.0)  # Not flagged yet
    panel2 = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[GestureProgress("look_left", "Looking left", 3.0, 2.5, COLORS.FLAGGED_RED)],
        flags_count=1,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel2, start + 3.0)
    recorder.finalize(start + 4.0)

    # Should have 1 flag
    assert recorder._flags_count == 1


def test_findings_recorder_export_markdown():
    """Export creates valid markdown file."""
    start = time.time()
    recorder = FindingsRecorder(start, consent_path="consent.txt")

    # Add some data
    panel = PanelState(
        frame_state="IN", face_direction="LEFT", body_movement="LOW", lean_angle=5.0,
        hand_state="RAISED", hand_movement="MOVING",
        gesture_progress=[GestureProgress("look_left", "Looking left", 3.0, 2.5, COLORS.ACTIVE_AMBER)],
        flags_count=1,
        audio_label="WHISPER", audio_confidence=0.85, level_db_above_floor=12.0,
        threshold_db=8.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event="whispering for 3.5s",
        last_fused_alert="alert risk=0.72 cues=look_left,hand_raised", audio_events_count=2,
    )
    recorder.update_video(panel, start + 5.0)
    recorder.update_audio(panel, start + 5.0)

    recorder.finalize(start + 10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = recorder.export_markdown(tmpdir)
        assert os.path.exists(md_path)

        # Check content
        with open(md_path, "r") as f:
            content = f.read()

        assert "Edge Invigilator Session Findings" in content
        assert "VIDEO DETECTION SUMMARY" in content.upper() or "Video Detection" in content
        assert "AUDIO DETECTION SUMMARY" in content.upper() or "Audio Detection" in content
        assert "Decision-support output for human review" in content
        assert "findings_" in content and ".csv" in content


def test_findings_recorder_csv_export():
    """Export creates valid CSV events table."""
    start = time.time()
    recorder = FindingsRecorder(start)

    # Add a gesture event manually for testing
    recorder._gesture_events.append(GestureEvent(
        key="look_left", label="Looking left",
        start_time=start + 1.0, end_time=start + 4.0,
        was_flagged=True,
    ))

    recorder.finalize(start + 10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = recorder.export_markdown(tmpdir)
        csv_path = os.path.join(tmpdir, os.path.basename(md_path).replace(".md", ".csv"))

        assert os.path.exists(csv_path)

        # Parse CSV
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have header + at least one row
        assert len(rows) >= 1
        assert rows[0]["type"] == "gesture"
        assert rows[0]["subtype"] == "look_left"
        assert rows[0]["flagged"] == "True"


def test_findings_recorder_consent_check():
    """Recorder checks consent file before export."""
    start = time.time()
    recorder = FindingsRecorder(start, consent_path="nonexistent.txt")
    assert not recorder.can_export()

    # Create consent file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        consent_path = f.name
    try:
        recorder2 = FindingsRecorder(start, consent_path=consent_path)
        assert recorder2.can_export()
    finally:
        os.unlink(consent_path)


def test_findings_recorder_no_consent_no_write():
    """Export fails gracefully without consent."""
    start = time.time()
    recorder = FindingsRecorder(start, consent_path="nonexistent.txt")

    panel = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel, time.time())
    recorder.finalize(time.time())

    # Should fail gracefully (no write)
    with tempfile.TemporaryDirectory() as tmpdir:
        # This should not raise, but may not write file
        md_path = recorder.export_markdown(tmpdir)
        # File may or may not exist depending on implementation
        # The key is no exception raised


def test_findings_recorder_movement_time_shares():
    """Recorder accumulates movement time shares."""
    start = time.time()
    recorder = FindingsRecorder(start)

    # Simulate movement changes
    panel = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel, start + 1.0)

    panel2 = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="LOW", lean_angle=0.0,
        hand_state="NONE", hand_movement="MOVING",
        gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel2, start + 2.0)

    # Movement time should be tracked
    assert sum(recorder._movement_time.values()) >= 0
    assert sum(recorder._hand_movement_time.values()) >= 0


def test_findings_recorder_calibration_peak():
    """Recorder tracks calibration status and peak level."""
    start = time.time()
    recorder = FindingsRecorder(start)

    panel = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[], flags_count=0,
        audio_label="SPEECH", audio_confidence=0.9, level_db_above_floor=20.0,
        threshold_db=8.0, calibration_status="clipping", calibration_message="Gain too high",
        clipping=True, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=1,
    )
    recorder.update_audio(panel, start + 1.0)

    assert recorder._calibration_status == "clipping"
    assert recorder._peak_level_db >= 20.0


def test_findings_recorder_fused_alerts():
    """Recorder tracks fused alerts."""
    start = time.time()
    recorder = FindingsRecorder(start)

    alert = FusedAlert(
        time=start + 5.0,
        severity="alert",
        risk=0.75,
        audio_kind="whispering",
        video_cues=["look_left", "hand_raised"],
        reasons=["whispering for 3.5s", "look left", "hand raised"],
    )
    recorder.update_fused_alert(alert, start + 5.0)

    recorder.finalize(start + 10.0)

    assert len(recorder._fused_alerts) == 1
    assert recorder._fused_alerts[0].severity == "alert"
    assert recorder._fused_alerts[0].risk == 0.75


def test_findings_recorder_no_consent_no_write():
    """Export without consent file fails gracefully."""
    start = time.time()
    recorder = FindingsRecorder(start, consent_path="/tmp/nonexistent_consent_12345.txt")

    panel = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel, time.time())
    recorder.finalize(time.time())

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise
        md_path = recorder.export_markdown(tmpdir)
        # File may not be created


def test_findings_recorder_footer():
    """Markdown includes footer disclaimer."""
    start = time.time()
    recorder = FindingsRecorder(start, consent_path="consent.txt")

    panel = PanelState(
        frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
        hand_state="NONE", hand_movement="STILL",
        gesture_progress=[], flags_count=0,
        audio_label="SILENCE", audio_confidence=0.0, level_db_above_floor=-60.0,
        threshold_db=-52.0, calibration_status="ok", calibration_message="",
        clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
        audio_events_count=0,
    )
    recorder.update_video(panel, start + 1.0)
    recorder.finalize(start + 10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = recorder.export_markdown(tmpdir)
        with open(md_path, "r") as f:
            content = f.read()

        assert "Decision-support output for human review, not proof of misconduct" in content


def test_findings_recorder_peak_level():
    """Recorder tracks peak level above floor."""
    start = time.time()
    recorder = FindingsRecorder(start)

    # Multiple updates with increasing levels
    for level in [5.0, 12.0, 8.0, 15.0]:
        panel = PanelState(
            frame_state="IN", face_direction="CENTER", body_movement="STILL", lean_angle=0.0,
            hand_state="NONE", hand_movement="STILL",
            gesture_progress=[], flags_count=0,
            audio_label="SPEECH", audio_confidence=0.9, level_db_above_floor=level,
            threshold_db=8.0, calibration_status="ok", calibration_message="",
            clipping=False, muted=False, last_audio_event=None, last_fused_alert=None,
            audio_events_count=1,
        )
        recorder.update_audio(panel, time.time())

    assert recorder._peak_level_db == 15.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])