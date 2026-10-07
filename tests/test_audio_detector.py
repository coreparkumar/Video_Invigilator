"""Tests for audio/detector.py - rule-based speech/whisper detector."""
import numpy as np
import pytest

from audio.detector import classify_window, DetectionResult
from audio.baseline import NoiseFloor
from audio.synth import ambient, speech, whisper, cough, steady_noise, paper_rustle
from audio.features import window_features
from audio.config import HOP_S


@pytest.fixture
def floor():
    """Calibrated NoiseFloor from 6s of ambient (seed=10)."""
    f = NoiseFloor()
    cal_signal = ambient(6.0, db=-62, seed=10)
    cal_feats = window_features(cal_signal)
    f.calibrate(cal_feats["frame_speech_db"])
    return f


@pytest.fixture
def base_ambient():
    """Short ambient for mixing with events."""
    return ambient(1.2, db=-62, seed=21)


def _run_detector(floor, signal):
    """Helper: run detector on signal."""
    feats = window_features(signal)
    return classify_window(feats, floor)


def test_ambient_only_silence(floor, base_ambient):
    """Ambient only -> silence."""
    result = _run_detector(floor, base_ambient)
    assert result.label == "silence"
    assert result.confidence == 1.0


def test_speech_detected(floor, base_ambient):
    """Speech -> speech label."""
    signal = base_ambient + speech(1.2, db=-35, f0=130)
    result = _run_detector(floor, signal)
    assert result.label == "speech"
    assert 0.0 <= result.confidence <= 1.0


def test_whisper_detected(floor, base_ambient):
    """Whisper -> whisper label."""
    signal = base_ambient + whisper(1.2, db=-38)
    result = _run_detector(floor, signal)
    assert result.label == "whisper"
    assert 0.0 <= result.confidence <= 1.0


def test_cough_noise_burst(floor, base_ambient):
    """Cough (short broadband burst) -> noise_burst."""
    # Cough at 0.5s in the 1.2s signal
    signal = base_ambient.copy()
    c = cough(0.25, db=-30)
    start = int(0.5 * 16000)
    end = min(start + len(c), len(signal))
    signal[start:end] += c[:end-start]
    result = _run_detector(floor, signal)
    assert result.label == "noise_burst"


def test_steady_noise_noise(floor, base_ambient):
    """Steady HVAC-like noise -> noise."""
    signal = base_ambient + steady_noise(1.2, db=-40)
    result = _run_detector(floor, signal)
    assert result.label == "noise"


def test_paper_rustle_noise(floor, base_ambient):
    """Paper rustle (bright) -> noise."""
    signal = base_ambient + paper_rustle(1.2, db=-38)
    result = _run_detector(floor, signal)
    assert result.label == "noise"


def test_near_miss_whisper_silence(floor, base_ambient):
    """Very quiet whisper below floor -> silence."""
    # Whisper at -68 dB, well below floor (~ -52 dB threshold)
    signal = base_ambient + whisper(1.2, db=-68)
    result = _run_detector(floor, signal)
    assert result.label == "silence"


def test_louder_whisper_higher_confidence(floor, base_ambient):
    """Louder whisper has higher confidence than quieter one, both labelled whisper."""
    w_loud = base_ambient + whisper(1.2, db=-35)
    w_quiet = base_ambient + whisper(1.2, db=-50)

    res_loud = _run_detector(floor, w_loud)
    res_quiet = _run_detector(floor, w_quiet)

    assert res_loud.label == "whisper"
    assert res_quiet.label == "whisper"
    assert res_loud.confidence > res_quiet.confidence


def test_confidence_in_range(floor, base_ambient):
    """Confidence always in [0, 1] for all signal types."""
    signals = {
        "silence": base_ambient,
        "speech": base_ambient + speech(1.2, db=-35),
        "whisper": base_ambient + whisper(1.2, db=-38),
        "noise_burst": base_ambient.copy(),
        "noise": base_ambient + steady_noise(1.2, db=-40),
        "rustle": base_ambient + paper_rustle(1.2, db=-38),
    }
    # Add cough at 0.5s for noise_burst
    c = cough(0.25, db=-30)
    start = int(0.5 * 16000)
    end = min(start + len(c), len(signals["noise_burst"]))
    signals["noise_burst"][start:end] += c[:end-start]

    for name, sig in signals.items():
        result = _run_detector(floor, sig)
        assert 0.0 <= result.confidence <= 1.0, f"{name}: confidence {result.confidence} out of range"


def test_active_frac_in_info(floor, base_ambient):
    """Info dict contains active_frac and level_db."""
    signal = base_ambient + speech(1.2, db=-35)
    result = _run_detector(floor, signal)
    assert "active_frac" in result.info
    assert "level_db" in result.info
    assert "above_floor_db" in result.info
    assert "level_std_db" in result.info
    assert "harmonicity" in result.info
    assert "centroid_hz" in result.info


def test_speech_harmonicity_high(floor, base_ambient):
    """Speech has high harmonicity in info."""
    signal = base_ambient + speech(1.2, db=-35, f0=130)
    result = _run_detector(floor, signal)
    assert result.info["harmonicity"] >= 0.45  # HARM_SPEECH


def test_whisper_low_harmonicity(floor, base_ambient):
    """Whisper has low harmonicity in info."""
    signal = base_ambient + whisper(1.2, db=-38)
    result = _run_detector(floor, signal)
    assert result.info["harmonicity"] < 0.45


def test_whisper_centroid_mid_range(floor, base_ambient):
    """Whisper centroid in 800-3500 Hz range."""
    signal = base_ambient + whisper(1.2, db=-38)
    result = _run_detector(floor, signal)
    assert 800 <= result.info["centroid_hz"] <= 3500


def test_rustle_centroid_high(floor, base_ambient):
    """Paper rustle has high centroid -> classified as noise."""
    signal = base_ambient + paper_rustle(1.2, db=-38)
    result = _run_detector(floor, signal)
    assert result.label == "noise"
    assert result.info["centroid_hz"] > 4200  # WHISPER_MAX_CENTROID_HZ


def test_cough_short_burst(floor, base_ambient):
    """Cough produces low active fraction -> noise_burst."""
    signal = base_ambient.copy()
    c = cough(0.25, db=-30)
    start = int(0.5 * 16000)
    end = min(start + len(c), len(signal))
    signal[start:end] += c[:end-start]
    result = _run_detector(floor, signal)
    assert result.label == "noise_burst"
    assert result.info["active_frac"] < 0.4  # ACTIVE_FRAC_SUSTAINED


def test_steady_noise_low_level_std(floor, base_ambient):
    """Steady noise has low level std -> classified as noise."""
    signal = base_ambient + steady_noise(1.2, db=-40)
    result = _run_detector(floor, signal)
    assert result.label == "noise"
    assert result.info["level_std_db"] < 2.5  # LEVEL_STD_MIN_DB


def test_result_dataclass_fields():
    """DetectionResult has correct fields."""
    result = DetectionResult("test", 0.5, {"key": "value"})
    assert result.label == "test"
    assert result.confidence == 0.5
    assert result.info == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])