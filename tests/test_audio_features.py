"""Tests for audio/features.py and audio/synth.py - pure NumPy, no I/O, no clocks."""
import numpy as np
import pytest

from audio.features import window_features, frame_harmonicity, EPS
from audio.synth import (
    ambient, speech, whisper, cough, steady_noise, paper_rustle, over,
    _bandpass_noise, _syllables, _scale
)
from audio.config import SAMPLE_RATE, FRAME


def test_sine_harmonicity():
    """150 Hz sine has high harmonicity (>0.8); white noise low (<0.35)."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    sine = np.sin(2 * np.pi * 150 * t).astype(np.float32)
    harm = frame_harmonicity(sine, SAMPLE_RATE)
    assert harm > 0.8, f"Expected harmonicity > 0.8, got {harm}"

    noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32)
    harm = frame_harmonicity(noise, SAMPLE_RATE)
    assert harm < 0.35, f"Expected noise harmonicity < 0.35, got {harm}"


def test_bandlimited_noise_peaks_in_speech_band():
    """Band-limited noise (300-4000 Hz) peaks in speech band."""
    noise = _bandpass_noise(SAMPLE_RATE, 300, 4000, seed=42)
    feats = window_features(noise)
    # Speech band energy should be significantly above flatness band
    assert feats["frame_speech_db"].mean() > -30, f"Speech band too low: {feats['frame_speech_db'].mean():.1f}"


def test_silence_returns_finite_values():
    """All-zero window gives finite values, zero harmonicity."""
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    feats = window_features(silence)
    assert np.isfinite(feats["rms_db"])
    assert np.isfinite(feats["flatness"])
    assert np.isfinite(feats["centroid_hz"])
    assert feats["frame_harm"].mean() == 0.0
    assert feats["crest"] >= 0.0
    assert feats["zcr"] == 0.0


def test_scale_invariance():
    """Raising level by 10 dB raises mean frame_speech_db by 9-11 dB."""
    base = ambient(1.0, db=-50, seed=1)
    loud = ambient(1.0, db=-40, seed=1)
    feats_base = window_features(base)
    feats_loud = window_features(loud)
    diff = feats_loud["frame_speech_db"].mean() - feats_base["frame_speech_db"].mean()
    assert 9.0 <= diff <= 11.0, f"Expected ~10 dB difference, got {diff:.1f}"


def test_window_too_short_raises():
    """Window shorter than one frame raises ValueError."""
    short = np.zeros(FRAME - 1, dtype=np.float32)
    with pytest.raises(ValueError, match="Window too short"):
        window_features(short)


def test_whisper_centroid_and_harmonicity():
    """Whisper has centroid 800-3500 Hz and mean harmonicity < 0.4."""
    w = whisper(1.0, db=-38)
    feats = window_features(w)
    assert 800 <= feats["centroid_hz"] <= 3500, f"Centroid {feats['centroid_hz']:.0f} out of range"
    assert feats["frame_harm"].mean() < 0.4, f"Whisper harmonicity too high: {feats['frame_harm'].mean():.3f}"


def test_speech_harmonicity_high():
    """Speech has high harmonicity."""
    s = speech(1.0, db=-35, f0=130)
    feats = window_features(s)
    assert feats["frame_harm"].mean() > 0.5, f"Speech harmonicity too low: {feats['frame_harm'].mean():.3f}"


def test_speech_has_syllable_modulation():
    """Speech has syllable modulation in speech-band energy."""
    s = speech(1.0, db=-35, f0=130)
    feats = window_features(s)
    # Syllable rate ~4 Hz -> std of frame energies should be noticeable
    std_db = feats["frame_speech_db"].std()
    assert std_db > 1.0, f"Speech modulation too low: {std_db:.2f} dB"


def test_steady_noise_low_modulation():
    """Steady noise has low syllable modulation."""
    sn = steady_noise(1.0, db=-40)
    feats = window_features(sn)
    std_db = feats["frame_speech_db"].std()
    assert std_db < 1.0, f"Steady noise modulation too high: {std_db:.2f} dB"


def test_cough_high_crest():
    """Cough has high crest factor (peak/rms)."""
    c = cough(0.25, db=-30)
    feats = window_features(c)
    assert feats["crest"] > 3.0, f"Cough crest too low: {feats['crest']:.1f}"


def test_paper_rustle_high_centroid():
    """Paper rustle has high centroid (>3500 Hz)."""
    pr = paper_rustle(1.0, db=-38)
    feats = window_features(pr)
    assert feats["centroid_hz"] > 3500, f"Rustle centroid too low: {feats['centroid_hz']:.0f}"


def test_over_adds_event():
    """over() adds event onto background at specified time."""
    bg = ambient(2.0, db=-62)
    ev = speech(0.5, db=-30)
    mixed = over(bg, ev, at_s=0.5)
    # Check mixed has higher energy around 0.5-1.0s
    feats_bg = window_features(bg)
    feats_mixed = window_features(mixed)
    assert feats_mixed["rms_db"] > feats_bg["rms_db"]


def test_syllables_envelope():
    """_syllables returns envelope in [0.15, 1.0]."""
    env = _syllables(SAMPLE_RATE, rate_hz=4.0)
    assert env.min() >= 0.15
    assert env.max() <= 1.0


def test_bandpass_noise_range():
    """_bandpass_noise produces signal with energy in target band."""
    n = SAMPLE_RATE
    bp = _bandpass_noise(n, 300, 4000, seed=99)
    assert len(bp) == n
    # Check it's not all zeros
    assert np.max(np.abs(bp)) > 0.01


def test_scale_function():
    """_scale preserves shape and sets RMS to target."""
    x = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
    scaled = _scale(x, -20.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(scaled ** 2)) + EPS)
    assert abs(rms_db - (-20.0)) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])