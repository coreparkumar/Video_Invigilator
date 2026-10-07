"""Streaming audio pipeline: manages buffering, windowing, calibration, and classification.

Pure logic - time is deterministic (sample-count based), no system clock.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from audio.config import (
    SAMPLE_RATE, FRAME, WINDOW_S, HOP_S,
    CALIBRATION_S, SPEECH_BAND
)
from audio.features import window_features
from audio.baseline import NoiseFloor
from audio.detector import classify_window, DetectionResult


@dataclass(frozen=True)
class WindowResult:
    """Result of processing one analysis window."""
    t: float                    # window end timestamp (seconds)
    label: str                  # "silence", "speech", "whisper", "noise_burst", "noise", "calibrating"
    confidence: float           # 0..1
    level_db: float             # mean speech-band dB
    info: dict                  # detector info dict


class AudioPipeline:
    """Streaming audio pipeline with windowed processing and auto-calibration."""

    def __init__(
        self,
        t0: float = 0.0,
        sr: int = SAMPLE_RATE,
        auto_calibrate: bool = True,
    ):
        self.sr = sr
        self.t0 = t0
        self.auto_calibrate = auto_calibrate

        # Buffer management
        self._window_samples = int(WINDOW_S * sr)
        self._hop_samples = int(HOP_S * sr)
        self._buffer = np.array([], dtype=np.float32)
        self._absolute_start_idx = 0  # absolute sample index of buffer[0]

        # Calibration state
        self._floor = NoiseFloor()
        self._calibrating = auto_calibrate  # Start calibrating if auto_calibrate=True
        self._calibration_frames: List[np.ndarray] = []
        self._calibration_hops_needed = int(CALIBRATION_S / HOP_S) if auto_calibrate else 0

        # Statistics
        self._total_hops = 0

    @property
    def floor(self) -> NoiseFloor:
        return self._floor

    @property
    def calibrating(self) -> bool:
        return self._calibrating

    def push(self, samples: np.ndarray) -> List[WindowResult]:
        """Push new samples into the pipeline.

        Returns a list of WindowResult for each completed hop.
        """
        if len(samples) == 0:
            return []

        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        results = []

        # Process complete hops
        while len(self._buffer) >= self._window_samples:
            # Extract window
            window = self._buffer[:self._window_samples]
            end_abs_idx = self._absolute_start_idx + self._window_samples
            t = self.t0 + end_abs_idx / self.sr

            # Extract features
            feats = window_features(window, self.sr)

            if self._calibrating:
                # Collect speech-band frames for calibration
                self._calibration_frames.append(feats["frame_speech_db"])
                result = WindowResult(
                    t=t,
                    label="calibrating",
                    confidence=0.0,
                    level_db=float(np.mean(feats["frame_speech_db"])),
                    info={"stage": "calibrating"},
                )
                self._calibration_hops_needed -= 1
                if self._calibration_hops_needed <= 0:
                    # Finish calibration
                    all_frames = np.concatenate(self._calibration_frames)
                    self._floor.calibrate(all_frames)
                    self._calibrating = False
                    self._calibration_frames = []
            else:
                # Classify
                det_result = classify_window(feats, self._floor)
                result = WindowResult(
                    t=t,
                    label=det_result.label,
                    confidence=det_result.confidence,
                    level_db=float(np.mean(feats["frame_speech_db"])),
                    info=det_result.info,
                )
                # Adapt floor
                self._floor.adapt(
                    feats["frame_speech_db"],
                    dt=HOP_S,
                    label=det_result.label,
                )

            results.append(result)
            self._total_hops += 1

            # Advance buffer by one hop
            self._buffer = self._buffer[self._hop_samples:]
            self._absolute_start_idx += self._hop_samples

        return results

    def start_calibration(self) -> None:
        """Manually trigger calibration (e.g., user presses 'n')."""
        self._calibrating = True
        self._calibration_frames = []
        self._calibration_hops_needed = int(CALIBRATION_S / HOP_S)

    def reset_calibration(self) -> None:
        """Reset floor and re-enter calibration mode."""
        self._floor.reset()
        self.start_calibration()

    def get_state(self) -> dict:
        """Return current pipeline state for UI/debugging."""
        return {
            "calibrating": self._calibrating,
            "calibrated": self._floor.calibrated,
            "floor_mean_db": self._floor.mean_db,
            "floor_threshold_db": self._floor.threshold_db,
            "total_hops": self._total_hops,
            "buffer_dur_s": len(self._buffer) / self.sr,
        }


# Re-export for convenience
from audio.config import HOP_S, WINDOW_S, SAMPLE_RATE, FRAME


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np
    from audio.synth import ambient, whisper, speech, steady_noise, paper_rustle

    print("Testing AudioPipeline...")

    # Build 6s test signal: 5s calibration + segments
    segments = [
        ("ambient", ambient(5.0, db=-62, seed=20)),  # 5s calibration
        ("silence", ambient(4.0, db=-62, seed=21)),
        ("whisper", whisper(6.0, db=-38)),
        ("silence", ambient(3.0, db=-62, seed=22)),
        ("speech", speech(6.0, db=-35)),
        ("silence", ambient(3.0, db=-62, seed=23)),
        ("noise", steady_noise(6.0, db=-40)),
        ("silence", ambient(3.0, db=-62, seed=24)),
        ("rustle", paper_rustle(6.0, db=-38)),
    ]

    # Concatenate
    full_signal = np.concatenate([s for _, s in segments])
    total_dur = len(full_signal) / 16000
    print(f"Total signal: {total_dur:.1f}s")

    # Run pipeline
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    results = pipe.push(full_signal)

    print(f"Processed {len(results)} windows ({len(results)*0.5:.1f}s at {0.5}s hop)")

    # Print results
    for i, r in enumerate(results):
        if i % 4 == 0 or r.label != "calibrating":
            print(f"  t={r.t:.1f}s: {r.label:12s} conf={r.confidence:.2f} level={r.level_db:.1f}dB")

    # Check calibration
    assert pipe.floor.calibrated
    print(f"\nCalibrated floor: mean={pipe.floor.mean_db:.1f}dB, thresh={pipe.floor.threshold_db:.1f}dB")

    # Check detection
    labels = [r.label for r in results if r.label != "calibrating"]
    assert "whisper" in labels
    assert "speech" in labels
    assert "noise" in labels
    print("Detection labels found:", set(labels))

    print("\nAudioPipeline smoke test passed!")