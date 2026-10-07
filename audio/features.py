"""Audio features: pure NumPy, no I/O, no clocks. Use constants from audio/config.py only."""
import numpy as np
from audio.config import (
    FRAME, SAMPLE_RATE, SPEECH_BAND, FLATNESS_BAND, PITCH_MIN_HZ, PITCH_MAX_HZ
)

EPS = 1e-12
_WIN = np.hanning(FRAME)


def _autocorr_peak_normalized(x: np.ndarray, sr: int, fmin: int, fmax: int) -> float:
    """Normalized autocorrelation peak in pitch range. Returns 0..1."""
    n = len(x)
    if n < 2:
        return 0.0
    # Remove DC
    x = x - x.mean()
    energy = np.dot(x, x)
    if energy < 1e-10:
        return 0.0
    # FFT-based autocorrelation (zero-pad to 2n)
    N = 1 << (2 * n - 1).bit_length()  # next power of 2 >= 2n
    X = np.fft.rfft(x, N)
    acf = np.fft.irfft(X * np.conj(X), N)[:n]
    acf = acf / (acf[0] + EPS)
    # Lag range for pitch
    lag_min = max(1, int(sr / fmax))
    lag_max = min(n - 1, int(sr / fmin))
    if lag_min >= lag_max:
        return 0.0
    # Bias correction: divide by (n - lag) / n
    lags = np.arange(lag_min, lag_max + 1)
    corrected = acf[lag_min:lag_max + 1] / ((n - lags) / n)
    peak = np.max(corrected)
    return float(np.clip(peak, 0.0, 1.0))


def frame_harmonicity(frame: np.ndarray, sr: int, fmin: int = PITCH_MIN_HZ, fmax: int = PITCH_MAX_HZ) -> float:
    """Harmonicity of a single frame via normalized autocorrelation peak."""
    return _autocorr_peak_normalized(frame, sr, fmin, fmax)


def window_features(window: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Compute features for a 1-second window (split into FRAME-sample frames)."""
    if len(window) < FRAME:
        raise ValueError(f"Window too short: {len(window)} samples, need at least {FRAME}")
    
    n_frames = len(window) // FRAME
    if n_frames == 0:
        raise ValueError("Window shorter than one frame")
    
    # Split into frames
    frames = window[:n_frames * FRAME].reshape(n_frames, FRAME)
    
    # Apply window and FFT
    windowed = frames * _WIN
    spec = np.fft.rfft(windowed, axis=1)
    # Normalize by window sum/2
    spec = spec / (_WIN.sum() / 2.0)
    power = np.abs(spec) ** 2
    
    # Frequency bins
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sr)
    
    # Band masks
    speech_lo, speech_hi = SPEECH_BAND
    flat_lo, flat_hi = FLATNESS_BAND
    
    speech_mask = (freqs >= speech_lo) & (freqs <= speech_hi)
    flat_mask = (freqs >= flat_lo) & (freqs <= flat_hi)
    
    # Per-frame speech-band energy (dB)
    frame_speech_power = np.sum(power[:, speech_mask], axis=1)
    frame_speech_db = 10.0 * np.log10(frame_speech_power + EPS)
    
    # Per-frame harmonicity
    frame_harm = np.array([frame_harmonicity(f, sr) for f in frames])
    
    # Window-averaged power spectrum for flatness/centroid
    avg_power = np.mean(power, axis=0)
    flat_power = avg_power[flat_mask]
    if len(flat_power) > 0 and np.sum(flat_power) > EPS:
        geo_mean = np.exp(np.mean(np.log(flat_power + EPS)))
        arith_mean = np.mean(flat_power)
        flatness = float(geo_mean / (arith_mean + EPS))
    else:
        flatness = 0.0
    
    # Spectral centroid over flatness band
    if len(flat_power) > 0 and np.sum(flat_power) > EPS:
        flat_freqs = freqs[flat_mask]
        centroid_hz = float(np.sum(flat_freqs * flat_power) / np.sum(flat_power))
    else:
        centroid_hz = 0.0
    
    # RMS dB and crest factor
    rms = float(np.sqrt(np.mean(window ** 2) + EPS))
    rms_db = 20.0 * np.log10(rms + EPS)
    peak = float(np.max(np.abs(window)))
    crest = peak / (rms + EPS)
    
    # Zero-crossing rate
    zcr = float(np.mean(np.diff(np.signbit(window))) if len(window) > 1 else 0.0)
    
    return {
        "frame_speech_db": frame_speech_db,
        "frame_harm": frame_harm,
        "flatness": flatness,
        "centroid_hz": centroid_hz,
        "rms_db": rms_db,
        "crest": crest,
        "zcr": zcr,
    }


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np
    from audio.config import SAMPLE_RATE
    
    # 150 Hz sine
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    sine = np.sin(2 * np.pi * 150 * t)
    feats = window_features(sine)
    print(f"150 Hz sine: harmonicity={feats['frame_harm'].mean():.3f}, flatness={feats['flatness']:.3f}")
    
    # White noise
    noise = np.random.randn(SAMPLE_RATE).astype(np.float32)
    feats = window_features(noise)
    print(f"White noise: harmonicity={feats['frame_harm'].mean():.3f}, flatness={feats['flatness']:.3f}")
    
    # Band-limited noise (speech band)
    from audio.synth import _bandpass_noise
    band_noise = _bandpass_noise(SAMPLE_RATE, 300, 4000, seed=42)
    feats = window_features(band_noise)
    print(f"Band noise 300-4000: speech_db={feats['frame_speech_db'].mean():.1f}")
    
    print("Smoke test passed")