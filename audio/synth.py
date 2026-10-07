"""Synthetic audio signals for testing. All seeded, return float32 arrays at 16 kHz."""
import numpy as np
from audio.config import SAMPLE_RATE, FRAME, SPEECH_BAND, FLATNESS_BAND, PITCH_MIN_HZ, PITCH_MAX_HZ


def _scale(x: np.ndarray, db: float) -> np.ndarray:
    """Scale signal to given RMS in dBFS."""
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    if rms < 1e-12:
        return x
    target = 10 ** (db / 20.0)
    return (x / rms * target).astype(np.float32)


def ambient(sec: float, db: float = -62, seed: int = 0) -> np.ndarray:
    """White noise at given dBFS."""
    rng = np.random.default_rng(seed)
    n = int(sec * SAMPLE_RATE)
    return _scale(rng.standard_normal(n), db)


def _bandpass_noise(n: int, lo: int, hi: int, seed: int = 0) -> np.ndarray:
    """FFT-mask noise, gentle spectral tilt."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    N = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, N)
    freqs = np.fft.rfftfreq(N, 1.0 / SAMPLE_RATE)
    mask = (freqs >= lo) & (freqs <= hi)
    # Gentle tilt: divide by sqrt(max(f,100)/100)
    tilt = np.sqrt(np.maximum(freqs, 100) / 100.0)
    X[mask] /= tilt[mask]
    X[~mask] = 0
    y = np.fft.irfft(X, N)[:n]
    return y.astype(np.float32)


def _syllables(n: int, rate_hz: float = 4.0) -> np.ndarray:
    """Syllable envelope: 0.15 + 0.85 * 0.5 * (1 + sin(2*pi*rate*t))."""
    t = np.arange(n) / SAMPLE_RATE
    env = 0.15 + 0.85 * 0.5 * (1 + np.sin(2 * np.pi * rate_hz * t))
    return env.astype(np.float32)


def speech(sec: float, db: float = -35, f0: int = 130) -> np.ndarray:
    """Harmonic speech-like signal with syllables and vibrato."""
    n = int(sec * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    # Base pitch with 3 Hz, +/-2% vibrato
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * f0 * vibrato * t
    # Sum harmonics k=1..max with 1/k amplitude
    max_k = int(3800 / f0)
    signal = np.zeros(n, dtype=np.float32)
    for k in range(1, max_k):
        signal += (1.0 / k) * np.sin(k * phase)
    # Apply syllable envelope
    signal *= _syllables(n, rate_hz=4.0)
    return _scale(signal, db)


def whisper(sec: float, db: float = -38) -> np.ndarray:
    """Whisper-like: bandpass noise 500-4500 Hz with syllable envelope."""
    n = int(sec * SAMPLE_RATE)
    noise = _bandpass_noise(n, 500, 4500)
    noise *= _syllables(n, rate_hz=4.0)
    return _scale(noise, db)


def cough(sec: float = 0.25, db: float = -30) -> np.ndarray:
    """Cough: white noise with fast exponential decay."""
    n = int(sec * SAMPLE_RATE)
    noise = np.random.default_rng(123).standard_normal(n)
    # Fast decay: exp(-t / (0.06 * sr))
    decay = np.exp(-np.arange(n) / (0.06 * SAMPLE_RATE))
    signal = noise * decay
    return _scale(signal, db)


def steady_noise(sec: float, db: float = -40) -> np.ndarray:
    """Steady white noise (HVAC-like)."""
    return ambient(sec, db, seed=42)


def paper_rustle(sec: float, db: float = -38) -> np.ndarray:
    """Paper rustle: bandpass 3500-8000 Hz with fast syllables (9 Hz)."""
    n = int(sec * SAMPLE_RATE)
    noise = _bandpass_noise(n, 3500, 8000)
    noise *= _syllables(n, rate_hz=9.0)
    return _scale(noise, db)


def over(background: np.ndarray, event: np.ndarray, at_s: float = 0.0) -> np.ndarray:
    """Add event onto a copy of background at given time (seconds)."""
    out = background.copy()
    start = int(at_s * SAMPLE_RATE)
    end = min(start + len(event), len(out))
    ev_len = end - start
    out[start:end] += event[:ev_len]
    return out


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np
    from audio.config import SAMPLE_RATE
    from audio.features import window_features
    
    print("Testing synthetic signals...")
    
    # Ambient
    a = ambient(1.0, db=-62, seed=0)
    feats = window_features(a)
    print(f"Ambient: rms_db={feats['rms_db']:.1f}, flatness={feats['flatness']:.3f}")
    
    # Speech
    s = speech(1.0, db=-35, f0=130)
    feats = window_features(s)
    print(f"Speech: rms_db={feats['rms_db']:.1f}, harmonicity={feats['frame_harm'].mean():.3f}")
    
    # Whisper
    w = whisper(1.0, db=-38)
    feats = window_features(w)
    print(f"Whisper: rms_db={feats['rms_db']:.1f}, harmonicity={feats['frame_harm'].mean():.3f}, centroid={feats['centroid_hz']:.0f}")
    
    # Cough
    c = cough(0.25, db=-30)
    feats = window_features(c)
    print(f"Cough: rms_db={feats['rms_db']:.1f}, crest={feats['crest']:.1f}")
    
    # Steady noise
    sn = steady_noise(1.0, db=-40)
    feats = window_features(sn)
    print(f"Steady: rms_db={feats['rms_db']:.1f}, flatness={feats['flatness']:.3f}")
    
    # Paper rustle
    pr = paper_rustle(1.0, db=-38)
    feats = window_features(pr)
    print(f"Rustle: rms_db={feats['rms_db']:.1f}, centroid={feats['centroid_hz']:.0f}")
    
    # Over
    bg = ambient(2.0, db=-62)
    ev = speech(0.5, db=-30)
    mixed = over(bg, ev, at_s=0.5)
    feats = window_features(mixed)
    print(f"Over: rms_db={feats['rms_db']:.1f}")
    
    print("All synthetic signals generated OK")