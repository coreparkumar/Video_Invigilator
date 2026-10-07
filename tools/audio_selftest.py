"""Synthetic audio self-test: runs pipeline on synthetic signals and validates detection.

Run: python tools/audio_selftest.py
Exit code 0 = PASS, 1 = FAIL
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from audio.pipeline import AudioPipeline
from audio.synth import (
    ambient, whisper, speech, steady_noise, paper_rustle
)
from audio.config import HOP_S, SAMPLE_RATE


def run_selftest() -> bool:
    """Run the self-test. Returns True if PASS, False if FAIL."""
    print("=" * 60)
    print("AUDIO SELF-TEST (synthetic signals)")
    print("=" * 60)

    # Build test signal: calibration + labeled segments
    segments = [
        ("calibration", ambient(5.0, db=-62, seed=20)),
        ("silence", ambient(4.0, db=-62, seed=21)),
        ("whisper", whisper(6.0, db=-38)),
        ("silence", ambient(3.0, db=-62, seed=22)),
        ("speech", speech(6.0, db=-35)),
        ("silence", ambient(3.0, db=-62, seed=23)),
        ("noise", steady_noise(6.0, db=-40)),
        ("silence", ambient(3.0, db=-62, seed=24)),
        ("rustle", paper_rustle(6.0, db=-38)),
    ]

    # Concatenate full signal
    full_signal = np.concatenate([s for _, s in segments])
    total_dur = len(full_signal) / SAMPLE_RATE

    print(f"Total signal duration: {total_dur:.1f}s")
    print(f"Sample rate: {SAMPLE_RATE} Hz, Hop: {HOP_S}s ({int(HOP_S*SAMPLE_RATE)} samples)")

    # Run pipeline
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)

    start_time = time.perf_counter()
    results = pipe.push(full_signal)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\nProcessed {len(results)} windows ({len(results)*HOP_S:.1f}s at {HOP_S}s hop)")
    print(f"Compute time: {elapsed_ms:.1f}ms ({elapsed_ms/len(results):.2f}ms per hop)")

    # Verify calibration happened
    assert pipe.floor.calibrated, "Floor should be calibrated"
    print(f"\nCalibrated floor: mean={pipe.floor.mean_db:.1f}dB, thresh={pipe.floor.threshold_db:.1f}dB")

    # Collect majority label per segment (ignore first 1.2s of each for boundary effects)
    segment_results = {}
    sample_pos = 0

    for seg_name, seg_signal in segments:
        seg_start_sample = sample_pos
        seg_end_sample = sample_pos + len(seg_signal)
        seg_start_time = seg_start_sample / SAMPLE_RATE
        seg_end_time = seg_end_sample / SAMPLE_RATE

        # Find windows in this segment (ignore first 1.2s = 2.4 hops to avoid boundary)
        boundary_skip = int(1.2 / HOP_S)  # 2-3 hops
        seg_windows = [
            r for r in results
            if seg_start_time + boundary_skip * HOP_S <= r.t <= seg_end_time
        ]

        if seg_windows:
            labels = [r.label for r in seg_windows]
            # Majority vote
            label_counts = {}
            for lbl in labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            majority = max(label_counts, key=label_counts.get)
            share = label_counts[majority] / len(labels)
            segment_results[seg_name] = (majority, share, labels)
        else:
            segment_results[seg_name] = ("none", 0.0, [])

        sample_pos = seg_end_sample

    # Expected majority labels
    expected = {
        "calibration": "calibrating",
        "silence": "silence",
        "whisper": "whisper",
        "speech": "speech",
        "noise": "noise",
        "rustle": "noise",  # paper rustle -> noise (bright)
    }

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Segment':<15} {'Expected':<12} {'Got':<12} {'Share':<8} {'PASS/FAIL'}")
    print("-" * 80)

    all_pass = True
    for seg_name, (majority, share, _) in segment_results.items():
        exp = expected.get(seg_name, "?")
        passed = (majority == exp) and (share >= 0.8)
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"{seg_name:<15} {exp:<12} {majority:<12} {share*100:>5.0f}%   {status}")

    print("=" * 80)

    # Compute performance
    compute_ms_per_hop = 1000.0 * sum(
        1 for _ in range(1)  # placeholder
    ) / max(1, len(results))  # simplified

    print(f"\nPerformance: {elapsed_ms/len(results):.2f}ms per hop (budget: 100ms)")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


if __name__ == "__main__":
    success = run_selftest()
    sys.exit(0 if success else 1)