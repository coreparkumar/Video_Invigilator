"""Tests for audio/baseline.py - noise-floor calibration with safety gate."""
import numpy as np
import pytest

from audio.baseline import NoiseFloor
from audio.config import HOP_S, MIN_MARGIN_DB, MARGIN_STD_MULT
from audio.synth import ambient, whisper
from audio.features import window_features


@pytest.fixture
def floor():
    """Calibrated NoiseFloor from 6s of ambient (seed=10)."""
    f = NoiseFloor()
    # 6s ambient at -62 dB
    signal = ambient(6.0, db=-62, seed=10)
    feats = window_features(signal)
    f.calibrate(feats["frame_speech_db"])
    return f


def test_calibration_sets_mean_std(floor):
    """Calibrated floor has mean around -62 dB and margin >= 8 dB."""
    assert floor.calibrated
    assert -65 <= floor.mean_db <= -59, f"Mean {floor.mean_db:.1f} out of range"
    assert floor.margin_db >= MIN_MARGIN_DB, f"Margin {floor.margin_db:.1f} < {MIN_MARGIN_DB}"
    assert floor.threshold_db == floor.mean_db + floor.margin_db


def test_calibration_from_windows():
    """calibrate_from_windows works with list of feature dicts."""
    f = NoiseFloor()
    windows = []
    for i in range(10):  # 10 windows = 5s at HOP_S=0.5
        signal = ambient(1.0, db=-60, seed=100 + i)
        windows.append(window_features(signal))
    f.calibrate_from_windows(windows)
    assert f.calibrated
    assert -63 <= f.mean_db <= -57


def test_calibrate_empty_raises():
    """Calibrating with empty array raises ValueError."""
    f = NoiseFloor()
    with pytest.raises(ValueError, match="empty"):
        f.calibrate(np.array([]))


def test_adapt_uncalibrated_noop():
    """Adapt on uncalibrated floor does nothing."""
    f = NoiseFloor()
    f.adapt(np.array([-60.0, -61.0]), HOP_S, "silence")
    assert not f.calibrated


def test_adapt_only_on_silence(floor):
    """Adapt only runs when label is 'silence'."""
    original_mean = floor.mean_db
    # Feed whisper label - should not adapt
    whisper_signal = whisper(1.0, db=-38)
    whisper_feats = window_features(whisper_signal)
    floor.adapt(whisper_feats["frame_speech_db"], HOP_S, "whisper")
    assert abs(floor.mean_db - original_mean) < 0.01, "Whisper label should not adapt"

    # Feed noise label - should not adapt
    from audio.synth import steady_noise
    noise_signal = steady_noise(1.0, db=-40)
    noise_feats = window_features(noise_signal)
    floor.adapt(noise_feats["frame_speech_db"], HOP_S, "noise")
    assert abs(floor.mean_db - original_mean) < 0.01, "Noise label should not adapt"

    # Feed silence - SHOULD adapt if within gate
    silence_signal = ambient(1.0, db=floor.mean_db + 1.0)  # slightly louder
    silence_feats = window_features(silence_signal)
    floor.adapt(silence_feats["frame_speech_db"], HOP_S, "silence")
    # Should have moved slightly toward the new level


def test_slow_drift_absorbed(floor):
    """+3 dB ambient drift absorbed after 5 simulated minutes (600 updates)."""
    original_mean = floor.mean_db
    # Simulate 5 minutes of +3 dB ambient
    for _ in range(600):
        # 0.5s chunk at -59 dB (original was ~-62)
        drift_signal = ambient(0.5, db=-59, seed=20)
        drift_feats = window_features(drift_signal)
        floor.adapt(drift_feats["frame_speech_db"], HOP_S, "silence")

    # Should have moved close to -59 dB
    assert abs(floor.mean_db - (-59)) < 1.5, f"Drift not absorbed: {floor.mean_db:.1f} vs -59"
    assert floor.mean_db > original_mean, "Floor should have risen"


def test_sustained_whisper_not_absorbed(floor):
    """Sustained whisper (label='whisper') never moves the floor."""
    original_mean = floor.mean_db
    original_margin = floor.margin_db

    for _ in range(100):  # 50s of whisper
        w = whisper(0.5, db=-38)
        w_feats = window_features(w)
        floor.adapt(w_feats["frame_speech_db"], HOP_S, "whisper")

    assert abs(floor.mean_db - original_mean) < 0.01, f"Whisper absorbed! {floor.mean_db:.3f} vs {original_mean:.3f}"
    assert abs(floor.margin_db - original_margin) < 0.01


def test_loud_sound_not_absorbed(floor):
    """Loud sound labeled as silence but outside gate is not absorbed."""
    original_mean = floor.mean_db
    # Very loud sound (30 dB above floor) but labeled "silence" - outside gate
    loud_signal = ambient(0.5, db=-20, seed=999)
    loud_feats = window_features(loud_signal)
    floor.adapt(loud_feats["frame_speech_db"], HOP_S, "silence")

    assert abs(floor.mean_db - original_mean) < 0.01, f"Loud sound absorbed despite gate! {floor.mean_db:.3f} vs {original_mean:.3f}"


def test_recalibration_resets(floor):
    """Recalibration completely replaces the floor."""
    original_mean = floor.mean_db
    # New calibration at different level
    new_signal = ambient(5.0, db=-50, seed=999)
    new_feats = window_features(new_signal)
    floor.calibrate(new_feats["frame_speech_db"])

    assert -53 <= floor.mean_db <= -47
    assert floor.mean_db != original_mean


def test_adapt_scalar_equivalent(floor):
    """adapt_scalar produces same result as adapt with single-frame array."""
    floor2 = NoiseFloor()
    # Copy calibration
    floor2._mean_db = floor.mean_db
    floor2._std_db = floor.std_db

    level = -58.0
    floor.adapt(np.array([level, level, level]), HOP_S, "silence")
    floor2.adapt_scalar(level, HOP_S, "silence")

    assert abs(floor.mean_db - floor2.mean_db) < 0.001


def test_reset_clears_calibration(floor):
    """Reset returns to uncalibrated state."""
    assert floor.calibrated
    floor.reset()
    assert not floor.calibrated
    assert floor.mean_db is None
    assert floor.std_db == 0.0


def test_margin_calculation(floor):
    """Margin is max of MIN_MARGIN_DB and MARGIN_STD_MULT * std."""
    expected = max(MIN_MARGIN_DB, MARGIN_STD_MULT * floor.std_db)
    assert abs(floor.margin_db - expected) < 0.01


def test_threshold_db(floor):
    """Threshold = mean + margin."""
    expected = floor.mean_db + floor.margin_db
    assert abs(floor.threshold_db - expected) < 0.01


def test_default_unconservative():
    """Uncalibrated floor returns conservative defaults."""
    f = NoiseFloor()
    assert not f.calibrated
    assert f.mean_db is None
    assert f.margin_db == MIN_MARGIN_DB
    assert f.threshold_db == -40.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])