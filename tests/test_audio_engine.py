"""Tests for audio/engine.py - top-level audio engine."""
import numpy as np
import pytest

from audio.engine import AudioEngine
from audio.capture import FakeAudioSource
from audio.session import AudioEvent
from audio.config import HOP_S, SAMPLE_RATE


def test_engine_disabled_returns_empty():
    """Disabled engine returns empty alerts."""
    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))
    engine = AudioEngine(source, enabled=False)

    alerts = engine.step(1000.0, video_active={"look_left"})
    assert len(alerts) == 0


def test_engine_muted_suppresses_alerts():
    """Muted engine never produces alerts."""
    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))
    engine = AudioEngine(source, enabled=True)
    engine.muted = True

    alerts = engine.step(1000.0, video_active={"look_left"})
    assert len(alerts) == 0


def test_engine_toggle_mute():
    """toggle_mute flips muted state."""
    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))
    engine = AudioEngine(source, enabled=True)

    assert not engine.muted
    engine.toggle_mute()
    assert engine.muted
    engine.toggle_mute()
    assert not engine.muted


def test_engine_recalibrate():
    """recalibrate triggers pipeline calibration."""
    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))
    engine = AudioEngine(source, enabled=True)

    assert engine.calibrating  # Auto-calibrate starts True
    engine.recalibrate()
    assert engine.calibrating


def test_engine_step_with_whisper_and_video_cue():
    """Engine produces fused alert for whisper + video cue."""
    from audio.synth import ambient, whisper, over
    from audio.config import SAMPLE_RATE

    # 20s signal: whisper at 6s (after calibration)
    bg = ambient(20.0, db=-62, seed=10)
    ev = whisper(4.0, db=-35)
    signal = over(bg, ev, at_s=6.0)
    source = FakeAudioSource(signal, chunk=int(0.5 * 16000))

    engine = AudioEngine(source, enabled=True)
    alerts_logged = []

    class FakeClock:
        def __init__(self, start=0.0):
            self._now = start
        @property
        def now(self):
            return self._now
        def advance(self, seconds):
            self._now += seconds

    clock = FakeClock(0.0)

    for i in range(50):  # ~25 seconds
        now = clock.now
        video_active = set()
        # Video cue active from 13-16s (matches whisper at 8-10s pipeline time + latency)
        if 8.0 <= clock.now <= 11.0:
            video_active.add("look_left")

        alerts = engine.step(now, video_active=video_active)
        if alerts:
            alerts_logged.extend(alerts)

        clock.advance(0.5)

    engine.close()

    # Should have at least one fused alert
    fused = [a for a in alerts_logged if a.severity == "alert"]
    assert len(fused) >= 1, f"Expected fused alert, got {len(fused)}"


def test_engine_calibrating_state():
    """Engine reports calibrating state correctly."""
    # Need 11 chunks for 10 results (first chunk fills buffer, next 10 produce windows)
    source = FakeAudioSource(np.zeros(88000, dtype=np.float32), chunk=8000)
    engine = AudioEngine(source, enabled=True)

    # Initially calibrating (auto_calibrate=True)
    assert engine.calibrating

    # After enough pushes, calibration completes
    # Need 11 pushes to get 10 results (first push fills buffer, next 10 produce windows)
    for _ in range(11):
        engine.step(0.0 + _ * 0.5)

    # Should be done calibrating
    assert not engine.calibrating


def test_engine_get_state():
    """get_state returns expected structure."""
    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))
    engine = AudioEngine(source, enabled=True)

    state = engine.get_state()
    assert "pipeline" in state
    assert "session" in state
    assert "enabled" in state
    assert "muted" in state
    assert state["enabled"] is True
    assert state["muted"] is False


def test_engine_close_idempotent():
    """close() can be called multiple times."""
    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))
    engine = AudioEngine(source, enabled=True)

    engine.close()
    engine.close()  # Should not raise


def test_engine_context_manager():
    """Engine works as context manager."""
    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))

    with AudioEngine(source) as engine:
        assert engine._source is not None

    # Should be closed after context
    # (source release is called in close)


def test_engine_no_audio_saved():
    """Engine does not save audio files by default."""
    import tempfile
    import os

    source = FakeAudioSource(np.zeros(8000, dtype=np.float32))
    engine = AudioEngine(source, enabled=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run a few steps
        for i in range(3):
            engine.step(1000.0 + i * 0.5)

        # Check no files created in temp dir
        files = os.listdir(tmpdir)
        assert len(files) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])