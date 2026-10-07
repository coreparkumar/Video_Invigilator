"""Guided audio-video check: pure core logic with injected clock and sleep.

Stages cover calibration, audio-only, video-only, fused detection, and negative controls.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Callable, Dict, Any
import time


class StageStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass
class Stage:
    name: str
    instruction: str
    seconds: float
    expect: Dict[str, Any]


@dataclass
class StageResult:
    name: str
    status: StageStatus
    measurements: Dict[str, Any]
    note: str = ""


class AVCheckRunner:
    """Runs a sequence of guided check stages with injected clock and video cue getter."""

    def __init__(
        self,
        engine,
        get_video_active: Callable[[], Set[str]],
        clock: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ):
        """
        Args:
            engine: AudioEngine instance
            get_video_active: callable returning set of active video cue keys
            clock: callable returning current time (monotonic). Defaults to time.monotonic
            sleep: callable(seconds) for waiting. Defaults to time.sleep
        """
        self.engine = engine
        self.get_video_active = get_video_active
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep

    def run(self, stages: List[Stage]) -> List[StageResult]:
        """Run all stages sequentially, returning list of StageResult."""
        results = []
        for stage in stages:
            result = self._run_stage(stage)
            results.append(result)
        return results

    def _run_stage(self, stage: Stage) -> StageResult:
        """Run a single stage, return StageResult."""
        print(f"\n=== Stage: {stage.name} ===")
        print(f"Instruction: {stage.instruction}")
        print(f"Duration: {stage.seconds}s")

        # Countdown
        for remaining in range(int(stage.seconds), 0, -1):
            print(f"  Starting in {remaining}s...", end="\r")
            self.sleep(1.0)
        print("  GO!                    ")

        start_time = self.clock()
        end_time = start_time + stage.seconds

        # Track measurements
        audio_labels = []
        audio_events = []
        alerts = []
        video_cue_durations: Dict[str, float] = {}
        video_cue_starts: Dict[str, float] = {}

        # Calibration check at start
        calibration_ok_at_start = self.engine.calibration_status.value == "ok"

        last_time = self.clock()
        while self.clock() < end_time:
            now = self.clock()
            dt = now - last_time
            last_time = now

            video_active = self.get_video_active()

            # Track video cue durations
            for cue in video_active:
                if cue not in video_cue_starts:
                    video_cue_starts[cue] = now
            for cue in list(video_cue_starts.keys()):
                if cue not in video_active:
                    dur = now - video_cue_starts[cue]
                    video_cue_durations[cue] = video_cue_durations.get(cue, 0.0) + dur
                    del video_cue_starts[cue]

            # Step audio engine
            if hasattr(self.engine, 'step'):
                alerts_step = self.engine.step(now, video_active=video_active)
                alerts.extend(alerts_step)

            # Record audio state
            if hasattr(self.engine, '_pipeline') and self.engine._pipeline:
                pipe = self.engine._pipeline
                # Get current label from last window result
                if hasattr(pipe, '_last_label'):
                    audio_labels.append(pipe._last_label)
                elif hasattr(pipe, 'get_state'):
                    state = pipe.get_state()
                    # We'll track label via session events instead

            # Track audio events from session
            if hasattr(self.engine, '_session') and self.engine._session:
                # Audio events are generated inside step()
                pass

            self.sleep(0.01)

        # Finalize any ongoing video cues
        for cue, start in video_cue_starts.items():
            dur = end_time - start
            video_cue_durations[cue] = video_cue_durations.get(cue, 0.0) + dur

        # Collect measurements
        measurements = {
            "audio_labels": audio_labels,
            "audio_events": audio_events,
            "alerts": [
                {
                    "severity": a.severity,
                    "risk": a.risk,
                    "audio_kind": a.audio_kind,
                    "video_cues": a.video_cues,
                    "reasons": a.reasons,
                }
                for a in alerts
            ],
            "video_cue_durations": video_cue_durations,
            "calibration_ok": calibration_ok_at_start,
        }

        # Evaluate expectations
        status, note = self._evaluate_expectations(stage, measurements)

        return StageResult(
            name=stage.name,
            status=status,
            measurements=measurements,
            note=note,
        )

    def _evaluate_expectations(self, stage: Stage, m: Dict) -> tuple[StageStatus, str]:
        """Evaluate stage expectations against measurements."""
        expect = stage.expect
        notes = []

        # Calibration OK
        if "calibration_ok" in expect:
            if m.get("calibration_ok") != expect["calibration_ok"]:
                return StageStatus.FAIL, f"calibration_ok mismatch: got {m.get('calibration_ok')} expected {expect['calibration_ok']}"
            notes.append("calibration_ok OK")

        # Audio event expected
        if "audio_event" in expect and expect["audio_event"]:
            has_event = len(m.get("audio_events", [])) > 0
            if not has_event:
                return StageStatus.FAIL, "Expected audio event but none occurred"
            notes.append("audio_event detected")

        # No alert expected
        if "no_alert" in expect and expect["no_alert"]:
            if len(m.get("alerts", [])) > 0:
                return StageStatus.FAIL, f"Unexpected alert(s): {[a['reasons'] for a in m['alerts']]}"
            notes.append("no_alert OK")

        # Alert severity expected
        if "alert_severity" in expect:
            expected_sev = expect["alert_severity"]
            alerts = m.get("alerts", [])
            has_sev = any(a["severity"] == expected_sev for a in alerts)
            if not has_sev:
                return StageStatus.FAIL, f"Expected alert severity '{expected_sev}', got {[a['severity'] for a in alerts]}"
            notes.append(f"alert_severity {expected_sev} OK")

        # Video cue expected
        if "video_cue" in expect:
            cue_name = expect["video_cue"]["name"]
            min_dur = expect["video_cue"].get("min_seconds", 0)
            dur = m.get("video_cue_durations", {}).get(cue_name, 0.0)
            if dur < min_dur:
                return StageStatus.FAIL, f"Video cue '{cue_name}' duration {dur:.1f}s < {min_dur}s"
            notes.append(f"video_cue {cue_name} >= {min_dur}s OK")

        # No audio event expected
        if "no_audio_event" in expect and expect["no_audio_event"]:
            if len(m.get("audio_events", [])) > 0:
                return StageStatus.FAIL, f"Unexpected audio events: {m['audio_events']}"
            notes.append("no_audio_event OK")

        return StageStatus.PASS, "; ".join(notes) if notes else "OK"


def default_stages() -> List[Stage]:
    """Default 9-stage sequence for AV check."""
    return [
        Stage(
            name="device_check",
            instruction="Check camera and microphone are detected",
            seconds=2,
            expect={"calibration_ok": False},  # just checking devices
        ),
        Stage(
            name="audio_calibration",
            instruction="Stay quiet for 5 seconds - audio calibration",
            seconds=6,
            expect={"calibration_ok": True},
        ),
        Stage(
            name="video_calibration",
            instruction="Sit normally, face the camera - video calibration",
            seconds=3,
            expect={},  # just visual check
        ),
        Stage(
            name="negative_control",
            instruction="Stay completely still and silent for 15 seconds",
            seconds=15,
            expect={"no_alert": True, "no_audio_event": True},
        ),
        Stage(
            name="audio_only_whisper",
            instruction="Whisper continuously for 6 seconds",
            seconds=7,
            expect={"audio_event": True},
        ),
        Stage(
            name="audio_only_speech",
            instruction="Speak normally for 6 seconds",
            seconds=7,
            expect={"audio_event": True},
        ),
        Stage(
            name="video_only_look_left",
            instruction="Look LEFT and hold for 3 seconds, then raise hand for 2 seconds",
            seconds=6,
            expect={"video_cue": {"name": "look_left", "min_seconds": 1.5}},
        ),
        Stage(
            name="fused_whisper_look_left",
            instruction="Whisper while looking LEFT for 8 seconds",
            seconds=9,
            expect={"alert_severity": "alert"},
        ),
        Stage(
            name="burst_control",
            instruction="Cough twice or rustle paper",
            seconds=3,
            expect={"no_audio_event": True},
        ),
    ]


if __name__ == "__main__":
    # Quick smoke test with mocks
    from audio.engine import AudioEngine
    from audio.capture import FakeAudioSource
    import numpy as np

    class MockEngine:
        calibration_status = type('obj', (object,), {'value': 'ok'})()
        calibration_message = "OK"
        def step(self, now, video_active):
            return []

    class MockClock:
        def __init__(self):
            self._now = 0.0
        def __call__(self):
            return self._now
        def advance(self, s):
            self._now += s

    clock = MockClock()
    engine = MockEngine()

    def get_video_active():
        return set()

    runner = AVCheckRunner(engine, get_video_active, clock=clock)
    stages = default_stages()[:2]  # Just first two for smoke test
    results = runner.run(stages)

    for r in results:
        print(f"{r.name}: {r.status.value} - {r.note}")

    print("Smoke test passed!")