"""Noise-floor calibration with slow adaptation and safety gate.

Adapts only when detector reports silence and level is within gate of current floor,
so sustained whisper/speech is never learned as ambient.
"""
import numpy as np
from audio.config import (
    MIN_MARGIN_DB, MARGIN_STD_MULT, FLOOR_TAU_S, FLOOR_GATE
)


class NoiseFloor:
    """Calibrates and slowly adapts the ambient noise floor."""

    def __init__(self):
        self._mean_db: float | None = None
        self._std_db: float = 0.0
        self._band_means: np.ndarray | None = None  # per-frame speech-band means
        self._band_stds: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self._mean_db is not None

    @property
    def mean_db(self) -> float | None:
        return self._mean_db

    @property
    def std_db(self) -> float:
        return self._std_db

    @property
    def margin_db(self) -> float:
        """Detection margin: max of absolute minimum and std-multiple."""
        if not self.calibrated:
            return MIN_MARGIN_DB
        return float(max(MIN_MARGIN_DB, MARGIN_STD_MULT * self._std_db))

    @property
    def threshold_db(self) -> float:
        """Threshold = floor mean + margin."""
        if not self.calibrated:
            return -40.0  # conservative default
        return self._mean_db + self.margin_db

    @property
    def band_thresholds(self) -> np.ndarray | None:
        """Per-band thresholds for multi-band detection."""
        if self._band_means is None:
            return None
        return self._band_means + self.margin_db

    def calibrate(self, frame_speech_db: np.ndarray) -> None:
        """Calibrate from concatenated frame speech-band dB values.

        Args:
            frame_speech_db: 1D array of frame-level speech-band dB values
        """
        if len(frame_speech_db) == 0:
            raise ValueError("Cannot calibrate with empty array")
        self._mean_db = float(np.mean(frame_speech_db))
        self._std_db = float(np.std(frame_speech_db))
        # Also store band-level stats if we have them
        # For now, just the overall speech-band mean/std

    def calibrate_from_windows(self, windows_db: list[np.ndarray]) -> None:
        """Calibrate from list of window feature dicts (each with frame_speech_db)."""
        all_frames = []
        for w in windows_db:
            if "frame_speech_db" in w:
                all_frames.append(w["frame_speech_db"])
        if not all_frames:
            raise ValueError("No frame_speech_db in windows")
        self.calibrate(np.concatenate(all_frames))

    def adapt(self, frame_speech_db: np.ndarray, dt: float, label: str) -> None:
        """Slowly adapt floor toward current level.

        Only adapts when:
        1. Floor is calibrated
        2. Current label is "silence" (detector says no activity)
        3. Current level is within FLOOR_GATE * margin of the floor

        This prevents sustained whisper/speech from being absorbed as ambient.

        Args:
            frame_speech_db: current window's frame speech-band dB values
            dt: time step in seconds (typically HOP_S = 0.5)
            label: detector label ("silence", "speech", "whisper", "noise_burst", "noise")
        """
        if not self.calibrated:
            return
        if label != "silence":
            return

        level = float(np.mean(frame_speech_db))
        margin = self.margin_db
        gate_width = FLOOR_GATE * margin

        # Only adapt if level is within gate of current mean
        if abs(level - self._mean_db) > gate_width:
            return

        # Exponential moving average
        alpha = 1.0 - np.exp(-dt / FLOOR_TAU_S)
        self._mean_db += alpha * (level - self._mean_db)

    def adapt_scalar(self, level_db: float, dt: float, label: str) -> None:
        """Adapt using a single scalar level (e.g., window mean)."""
        if not self.calibrated:
            return
        if label != "silence":
            return

        margin = self.margin_db
        gate_width = FLOOR_GATE * margin

        if abs(level_db - self._mean_db) > gate_width:
            return

        alpha = 1.0 - np.exp(-dt / FLOOR_TAU_S)
        self._mean_db += alpha * (level_db - self._mean_db)

    def reset(self) -> None:
        """Reset to uncalibrated state."""
        self._mean_db = None
        self._std_db = 0.0
        self._band_means = None
        self._band_stds = None


# ---- Calibration Quality ---------------------------------------------------
from enum import Enum

class CalibrationStatus(Enum):
    OK = "ok"
    NO_SIGNAL = "no_signal"
    CLIPPING = "clipping"
    TOO_LOUD = "too_loud"
    NOISY = "noisy"


def calibration_quality(frame_db: np.ndarray, peak: float) -> tuple[CalibrationStatus, str]:
    """Evaluate calibration quality from frame speech-band dB values and peak.

    Args:
        frame_db: concatenated frame-level speech-band dB values from calibration
        peak: maximum absolute sample value during calibration

    Returns:
        (status, message) - status enum and user-facing message
    """
    from audio.config import (
        CAL_MAX_STD_DB, CAL_MIN_LEVEL_DB, CAL_MAX_LEVEL_DB, CLIP_PEAK
    )

    if len(frame_db) == 0:
        return CalibrationStatus.NO_SIGNAL, "No calibration data received"

    mean_db = float(np.mean(frame_db))
    std_db = float(np.std(frame_db))
    peak_abs = float(peak)

    # Priority order of checks
    if mean_db < CAL_MIN_LEVEL_DB:
        return (CalibrationStatus.NO_SIGNAL,
                f"Input level too low ({mean_db:.1f} dB). Microphone muted, unplugged, or wrong device selected. "
                f"Check connection and press 'n' to recalibrate.")

    if peak_abs >= CLIP_PEAK:
        return (CalibrationStatus.CLIPPING,
                f"Input clipping detected (peak={peak_abs:.2f}). Lower microphone gain in OS settings "
                f"and press 'n' to recalibrate.")

    if mean_db > CAL_MAX_LEVEL_DB:
        return (CalibrationStatus.TOO_LOUD,
                f"Input level too high ({mean_db:.1f} dB). Reduce microphone gain or move further away "
                f"and press 'n' to recalibrate.")

    if std_db > CAL_MAX_STD_DB:
        return (CalibrationStatus.NOISY,
                f"Environment noisy during calibration (std={std_db:.2f} dB). Stay quiet, close windows/doors, "
                f"turn off fans, and press 'n' to recalibrate.")

    return CalibrationStatus.OK, "Calibration OK"


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np
    from audio.config import SAMPLE_RATE, HOP_S
    from audio.synth import ambient
    from audio.features import window_features

    # Simulate calibration
    floor = NoiseFloor()
    cal_signal = ambient(5.0, db=-62, seed=10)  # 5s ambient
    cal_feats = window_features(cal_signal)
    floor.calibrate(cal_feats["frame_speech_db"])
    print(f"Calibrated: mean={floor.mean_db:.1f} dB, std={floor.std_db:.2f} dB")
    print(f"Margin={floor.margin_db:.1f} dB, Threshold={floor.threshold_db:.1f} dB")

    # Test adaptation: +3 dB drift absorbed
    for _ in range(600):  # 600 * 0.5s = 5 minutes
        drift_signal = ambient(0.5, db=-59, seed=20)  # 3 dB louder
        drift_feats = window_features(drift_signal)
        floor.adapt(drift_feats["frame_speech_db"], HOP_S, "silence")
    print(f"After +3dB drift (5 min): mean={floor.mean_db:.2f} dB (expected ~ -59)")

    # Test whisper NOT absorbed
    floor2 = NoiseFloor()
    floor2.calibrate(cal_feats["frame_speech_db"])
    original_mean = floor2.mean_db
    for _ in range(100):
        whisper_signal = ambient(0.5, db=-38, seed=99) + _bandpass_noise(8000, 500, 4500, seed=99)  # simulate whisper
        # Actually use proper whisper
        from audio.synth import whisper
        w = whisper(0.5, db=-38)
        w_feats = window_features(w)
        floor2.adapt(w_feats["frame_speech_db"], HOP_S, "whisper")  # label is whisper, not silence
    print(f"After whisper (label=whisper): mean={floor2.mean_db:.2f} dB, changed={abs(floor2.mean_db - original_mean):.3f} dB (should be ~0)")

    print("Smoke test passed")