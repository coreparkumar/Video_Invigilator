"""Tests for audio/pipeline.py - streaming pipeline with calibration and classification."""
import numpy as np
import pytest

from audio.pipeline import AudioPipeline, WindowResult
from audio.synth import ambient, whisper, speech, steady_noise, paper_rustle, over
from audio.config import SAMPLE_RATE, HOP_S, WINDOW_S, FRAME, CALIBRATION_S


def test_pipeline_first_windows_are_calibrating():
    """First CALIBRATION_S/HOP_S windows should be 'calibrating'."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)

    # Generate exactly 5s calibration + 1s = 6s at 16000 Hz = 96000 samples
    signal = ambient(6.0, db=-62, seed=10)
    results = pipe.push(signal)

    # First 10 hops (5s / 0.5s) should be calibrating
    calibrating_count = sum(1 for r in results if r.label == "calibrating")
    assert calibrating_count == int(CALIBRATION_S / HOP_S)  # 10

    # Rest should be classified
    classified = [r for r in results if r.label != "calibrating"]
    assert all(r.label in ("silence", "speech", "whisper", "noise_burst", "noise") for r in classified)


def test_pipeline_consecutive_timestamps():
    """Window timestamps are 0.5s apart starting at t0 + 1.0s."""
    pipe = AudioPipeline(t0=100.0, auto_calibrate=False)
    pipe.floor.calibrate(np.array([-60.0] * 100))  # fake calibration

    # Push enough for 5 windows
    signal = ambient(3.0, db=-62, seed=1)
    results = pipe.push(signal)

    # Timestamps: t0 + WINDOW_S, t0 + WINDOW_S + HOP_S, ...
    expected_times = [100.0 + WINDOW_S + i * HOP_S for i in range(len(results))]
    for r, expected_t in zip(results, expected_times):
        assert abs(r.t - expected_t) < 0.01, f"Expected {expected_t:.2f}, got {r.t:.2f}"


def test_pipeline_chunked_feeding_same_as_bulk():
    """Feeding in small chunks gives same labels as bulk."""
    pipe1 = AudioPipeline(t0=0.0, auto_calibrate=False)
    pipe1.floor.calibrate(np.array([-60.0] * 100))

    pipe2 = AudioPipeline(t0=0.0, auto_calibrate=False)
    pipe2.floor.calibrate(np.array([-60.0] * 100))

    # 3s ambient with whisper overlaid at 1.0s for 1s
    bg = ambient(3.0, db=-62, seed=5)
    ev = whisper(1.0, db=-35)
    signal = over(bg, ev, at_s=1.0)

    # Bulk
    results1 = pipe1.push(signal)

    # Chunked: 700 samples at a time (less than one hop)
    chunk_size = 700
    results2 = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        results2.extend(pipe2.push(chunk))

    # Labels should match
    labels1 = [r.label for r in results1]
    labels2 = [r.label for r in results2]
    assert labels1 == labels2, f"Chunked differs: {labels1} vs {labels2}"


def test_pipeline_start_calibration():
    """Manual start_calibration triggers calibration mode."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=False)
    pipe.floor.calibrate(np.array([-60.0] * 100))

    assert not pipe.calibrating
    pipe.start_calibration()
    assert pipe.calibrating
    assert pipe._calibration_hops_needed == int(CALIBRATION_S / HOP_S)


def test_pipeline_reset_calibration():
    """reset_calibration resets floor and restarts calibration."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    # Need 6s for calibration (1s window + 10 hops * 0.5s = 6s)
    signal = ambient(6.0, db=-62, seed=10)
    pipe.push(signal)

    assert pipe.floor.calibrated
    original_mean = pipe.floor.mean_db

    # Reset and recalibrate at different level
    pipe.reset_calibration()
    assert pipe.calibrating

    # Feed new calibration data (6s)
    new_signal = ambient(6.0, db=-50, seed=999)
    pipe.push(new_signal)

    assert not pipe.calibrating
    assert pipe.floor.calibrated
    assert pipe.floor.mean_db != original_mean


def test_pipeline_get_state():
    """get_state returns expected fields."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    signal = ambient(2.0, db=-62, seed=1)
    pipe.push(signal)

    state = pipe.get_state()
    assert "calibrating" in state
    assert "calibrated" in state
    assert "floor_mean_db" in state
    assert "floor_threshold_db" in state
    assert "total_hops" in state
    assert "buffer_dur_s" in state


def test_pipeline_whisper_detection():
    """Pipeline detects whisper in mixed signal."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)

    # 6s calibration + 4s whisper overlaid on 4s ambient + 2s ambient
    bg = ambient(12.0, db=-62, seed=10)
    ev = whisper(4.0, db=-38)
    signal = over(bg, ev, at_s=6.0)  # whisper starts at 6s (after calibration)
    results = pipe.push(signal)

    labels = [r.label for r in results if r.label != "calibrating"]
    assert "whisper" in labels
    whisper_count = sum(1 for r in results if r.label == "whisper")
    silence_count = sum(1 for r in results if r.label == "silence")
    assert whisper_count > silence_count


def test_pipeline_speech_detection():
    """Pipeline detects speech."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)

    bg = ambient(12.0, db=-62, seed=10)
    ev = speech(4.0, db=-35)
    signal = over(bg, ev, at_s=6.0)
    results = pipe.push(signal)

    labels = [r.label for r in results if r.label != "calibrating"]
    assert "speech" in labels


def test_pipeline_noise_detection():
    """Pipeline detects steady noise and rustle as noise."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)

    bg = ambient(12.0, db=-62, seed=10)
    ev = steady_noise(4.0, db=-40)
    signal = over(bg, ev, at_s=6.0)
    results = pipe.push(signal)

    labels = [r.label for r in results if r.label != "calibrating"]
    assert "noise" in labels


def test_pipeline_rustle_detection():
    """Paper rustle detected as noise (bright)."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)

    bg = ambient(12.0, db=-62, seed=10)
    ev = paper_rustle(4.0, db=-38)
    signal = over(bg, ev, at_s=6.0)
    results = pipe.push(signal)

    labels = [r.label for r in results if r.label != "calibrating"]
    assert "noise" in labels


def test_pipeline_window_result_fields():
    """WindowResult has all required fields."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=False)
    pipe.floor.calibrate(np.array([-60.0] * 100))

    signal = ambient(1.0, db=-62, seed=1)
    results = pipe.push(signal)

    assert len(results) > 0
    for r in results:
        assert isinstance(r, WindowResult)
        assert isinstance(r.t, float)
        assert isinstance(r.label, str)
        assert isinstance(r.confidence, float)
        assert isinstance(r.level_db, float)
        assert isinstance(r.info, dict)


def test_pipeline_empty_push():
    """Empty push returns empty list."""
    pipe = AudioPipeline(t0=0.0)
    results = pipe.push(np.array([], dtype=np.float32))
    assert results == []


def test_pipeline_partial_window():
    """Pushing less than one window returns no results until window is full."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=False)
    pipe.floor.calibrate(np.array([-60.0] * 100))

    # Push half a window
    half_window = ambient(0.5, db=-62, seed=1)
    results = pipe.push(half_window)
    assert results == []

    # Push another half to complete window
    results = pipe.push(half_window)
    assert len(results) == 1


def test_pipeline_no_files_created(tmp_path):
    """Pipeline creates no files during operation."""
    import os

    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    signal = ambient(2.0, db=-62, seed=1)
    results = pipe.push(signal)

    # Check no new files in tmp_path (use chdir)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pipe.push(signal)
        files = list(tmp_path.glob("*"))
        assert len(files) == 0
    finally:
        os.chdir(old_cwd)


def test_pipeline_calibration_status_ok():
    """Pipeline reports OK calibration status for normal ambient."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    signal = ambient(6.0, db=-62, seed=10)
    pipe.push(signal)
    
    assert pipe.calibration_status.value == "ok"
    assert "OK" in pipe.calibration_message or "ok" in pipe.calibration_message.lower()


def test_pipeline_calibration_status_no_signal():
    """Pipeline detects NO_SIGNAL for muted/dead mic."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    signal = np.zeros(88000, dtype=np.float32)  # 5.5s silence
    pipe.push(signal)
    
    assert pipe.calibration_status.value == "no_signal"
    assert "low" in pipe.calibration_message.lower() or "muted" in pipe.calibration_message.lower()


def test_pipeline_calibration_status_clipping():
    """Pipeline detects CLIPPING for overdriven input."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    # Full-scale signal (0 dBFS) for 5.5s
    signal = np.ones(88000, dtype=np.float32) * 1.0
    pipe.push(signal)
    
    assert pipe.calibration_status.value == "clipping"
    assert "clipping" in pipe.calibration_message.lower() or "gain" in pipe.calibration_message.lower()


def test_pipeline_input_peak_and_clipping():
    """Pipeline tracks input peak and detects clipping."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=False)
    pipe.floor.calibrate(np.array([-60.0] * 100))
    
    # Normal signal - no clipping
    normal = np.random.default_rng(1).normal(0, 0.1, 8000).astype(np.float32)
    pipe.push(normal)
    assert not pipe.clipping
    assert pipe.input_peak < 0.5
    
    # Clipped signal
    clipped = np.ones(8000, dtype=np.float32) * 1.0
    pipe.push(clipped)
    assert pipe.clipping
    assert pipe.input_peak >= 0.98


def test_pipeline_calibration_status_in_get_state():
    """get_state includes calibration status and message."""
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    signal = ambient(6.0, db=-62, seed=10)
    pipe.push(signal)
    
    state = pipe.get_state()
    assert "calibration_status" in state
    assert "calibration_message" in state
    assert state["calibration_status"] == "ok"
    assert "input_peak" in state
    assert "clipping" in state
    assert state["clipping"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])