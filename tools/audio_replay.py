"""Audio replay tool: offline analysis of recorded WAV files.

Loads a WAV file, runs the audio pipeline + session offline, and prints
a timeline with labels, confidence, dB above floor, and EVENT markers.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

from audio.capture import load_wav
from audio.pipeline import AudioPipeline
from audio.session import AudioSession
from audio.config import HOP_S, SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser(description="Replay and analyze a WAV file offline")
    ap.add_argument("wav_file", help="Path to WAV file")
    ap.add_argument("--list-only", action="store_true", help="Only list events, no timeline")
    args = ap.parse_args()

    wav_path = Path(args.wav_file)
    if not wav_path.exists():
        print(f"Error: {wav_path} does not exist")
        sys.exit(1)

    print(f"Loading {wav_path}...")
    signal = load_wav(str(wav_path), sr=SAMPLE_RATE)
    dur = len(signal) / SAMPLE_RATE
    print(f"Duration: {dur:.1f}s, Sample rate: {SAMPLE_RATE} Hz")

    # Warn if shorter than calibration period
    from audio.config import CALIBRATION_S
    if dur < CALIBRATION_S:
        print(f"WARNING: File shorter than calibration period ({CALIBRATION_S}s). Calibration may be unreliable.")

    # Run pipeline
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    results = pipe.push(np.asarray(signal, dtype=np.float32))

    # Run session
    session = AudioSession()
    events = []
    for r in results:
        evs = session.update(r.label, r.confidence, r.t)
        events.extend(evs)

    # Print timeline
    print("\n" + "=" * 100)
    print(f"{'Time':>8} {'Label':<15} {'Conf':>6} {'Level':>8} {'AboveFl':>8} {'Event'}")
    print("-" * 100)

    for r in results:
        marker = ""
        if r.label == "calibrating":
            marker = "CAL"
        elif r.label in ("speech", "whisper"):
            marker = "ACT"
        event_str = ""
        for ev in events:
            if abs(ev.start - r.t) < 0.01:
                event_str = f"EVENT: {ev.kind} ({ev.duration:.1f}s)"
                break
        print(f"{r.t:8.2f} {r.label:<15} {r.confidence:6.2f} {r.level_db:8.1f} "
              f"{r.info.get('above_floor_db', 0):8.1f} {event_str}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY:")
    labels = [r.label for r in results if r.label != "calibrating"]
    from collections import Counter
    counts = Counter(labels)
    for label, count in counts.most_common():
        print(f"  {label}: {count} windows ({count * HOP_S:.1f}s)")
    print(f"Total audio events: {len(events)}")
    for ev in events:
        print(f"  {ev.kind}: {ev.duration:.1f}s, conf={ev.mean_conf:.2f}, {ev.start:.1f}-{ev.end:.1f}s")


if __name__ == "__main__":
    main()