"""Audio engine: ties together pipeline, session, and fusion for live integration."""
from typing import List, Optional, Set
import numpy as np

from audio.pipeline import AudioPipeline
from audio.session import AudioSession
from audio.fusion import Fusion, Alert
from audio.config import HOP_S, AUDIO_HOLD_S, CLIP_PEAK
from audio.capture import AudioSource
from audio.baseline import CalibrationStatus


class AudioEngine:
    """Top-level audio engine for live operation.

    Owns an AudioPipeline, AudioSession, and Fusion.
    """

    def __init__(
        self,
        source: AudioSource,
        t0: float = 0.0,
        enabled: bool = True,
    ):
        self._source = source
        self._enabled = enabled
        self._pipeline = AudioPipeline(t0=0.0, auto_calibrate=True)
        self._session = AudioSession()
        self._fusion = Fusion()

    @property
    def calibrating(self) -> bool:
        return self._pipeline.calibrating

    @property
    def calibration_status(self) -> CalibrationStatus:
        return self._pipeline.calibration_status

    @property
    def calibration_message(self) -> str:
        return self._pipeline.calibration_message

    @property
    def input_peak(self) -> float:
        return self._pipeline.input_peak

    @property
    def clipping(self) -> bool:
        return self._pipeline.clipping

    @property
    def muted(self) -> bool:
        return self._session.muted

    @muted.setter
    def muted(self, value: bool) -> None:
        self._session.muted = value

    def toggle_mute(self) -> None:
        """Toggle mute state."""
        self._session.muted = not self._session.muted

    def recalibrate(self) -> None:
        """Trigger noise-floor recalibration."""
        self._pipeline.start_calibration()

    def step(
        self,
        now: float,
        video_active: Optional[Set[str]] = None,
        lip_activity: float = 0.0,
    ) -> List[Alert]:
        """Process one audio step.

        Args:
            now: current timestamp (monotonic)
            video_active: set of active video gesture keys from InvigilatorSession
            lip_activity: 0..1 lip activity score from LipTracker

        Returns:
            List of Alert (0 or more)
        """
        if not self._enabled:
            return []

        # Read audio frames
        samples = self._source.read()
        if len(samples) == 0:
            return []

        # Process through pipeline
        pipeline_results = self._pipeline.push(samples)

        # Collect audio events from session
        audio_events = []
        for pr in pipeline_results:
            # Feed detector label into session
            events = self._session.update(pr.label, pr.confidence, pr.t)
            audio_events.extend(events)

        # Fuse with video context
        if video_active is None:
            video_active = set()

        alerts = self._fusion.update(audio_events, video_active, now, 0.0)
        return alerts

    def get_state(self) -> dict:
        """Return combined state for UI/debugging."""
        pipeline_state = self._pipeline.get_state()
        session_view = self._session.get_view()
        return {
            "pipeline": pipeline_state,
            "session": session_view,
            "enabled": self._enabled,
            "muted": self._session.muted,
        }

    def close(self) -> None:
        """Release resources."""
        if self._source:
            self._source.release()
            self._source = None
        # Pipeline, session, fusion have no external resources to close

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Quick smoke test with FakeAudioSource
    from audio.capture import FakeAudioSource
    from audio.session import AudioEvent
    from audio.config import HOP_S

    class FakeClock:
        def __init__(self, start=1000.0):
            self._now = start
        @property
        def now(self):
            return self._now
        def advance(self, seconds):
            self._now += seconds

    import numpy as np
    from audio.synth import ambient, whisper
    from audio.config import SAMPLE_RATE

    print("Testing AudioEngine...")

    # Create fake source with whisper event
    bg = ambient(20.0, db=-62, seed=10)
    ev = whisper(4.0, db=-35)
    from audio.synth import over
    signal = over(bg, ev, at_s=6.0)  # whisper at 6s
    source = FakeAudioSource(signal, chunk=int(0.5 * 16000))  # 0.5s chunks

    clock = FakeClock(1000.0)
    engine = AudioEngine(source, t0=clock.now, enabled=True)

    print("Running engine steps...")
    alerts_logged = []

    for i in range(50):  # ~25 seconds at 0.5s per step
        now = clock.now
        # Simulate video cues: look_left active from 13-16s
        video_active = set()
        if 13.0 <= (clock.now - 1000.0) <= 16.0:
            video_active.add("look_left")

        alerts = engine.step(now, video_active=video_active)
        if alerts:
            for a in alerts:
                print(f"  t={now:.1f}: {a.severity} risk={a.risk:.2f} kind={a.audio_kind} cues={a.video_cues}")
                alerts_logged.append(a)

        clock.advance(0.5)

    engine.close()

    # Check we got at least one fused alert
    fused = [a for a in alerts_logged if a.severity == "alert"]
    assert len(fused) >= 1, f"Expected at least 1 fused alert, got {len(fused)}"
    print(f"Logged {len(alerts_logged)} alerts, {len(fused)} fused")

    # Check calibration happened
    state = engine.get_state()
    assert state["pipeline"]["calibrated"]
    print(f"Pipeline calibrated: floor={state['pipeline']['floor_mean_db']:.1f}dB")

    print("AudioEngine smoke test passed!")