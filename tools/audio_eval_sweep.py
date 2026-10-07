"""Audio evaluation sweep: batch process a manifest of WAV files with ground truth.

Manifest CSV columns: wav, expected, distance_m, noise, consent_ok, notes
Refuses rows where consent_ok is not "yes".
Each WAV must start with 5s of quiet for calibration.
Outputs markdown table to docs/AUDIO_EVAL.md.
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np

from audio.capture import load_wav
from audio.pipeline import AudioPipeline
from audio.session import AudioSession
from audio.config import SAMPLE_RATE, HOP_S, CALIBRATION_S


def majority_label(labels):
    """Return majority label and its share."""
    from collections import Counter
    if not labels:
        return "none", 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    maj = max(counts, key=counts.get)
    return maj, counts[maj] / len(labels)


def main():
    ap = argparse.ArgumentParser(description="Batch evaluate WAV files against ground truth")
    ap.add_argument("manifest", help="CSV manifest: wav,expected,distance_m,noise,consent_ok,notes")
    ap.add_argument("--out", default="docs/AUDIO_EVAL.md", help="Output markdown file")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest {manifest_path} not found")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("consent_ok", "").lower() != "yes":
                print(f"Skipping {row['wav']}: consent_ok != 'yes'")
                continue

            wav_path = Path(row["wav"])
            if not wav_path.exists():
                print(f"  WARNING: {wav_path} not found, skipping")
                continue

            # Load WAV
            try:
                signal = load_wav(str(wav_path), sr=16000)
            except Exception as e:
                print(f"  ERROR loading {wav_path}: {e}")
                continue

            # Verify calibration prefix
            dur = len(signal) / 16000
            if dur < CALIBRATION_S:
                print(f"  WARNING: {wav_path} shorter than calibration period ({CALIBRATION_S}s)")

            # Run pipeline
            pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
            results = pipe.push(signal)

            # Collect labels after calibration (ignore first CALIBRATION_S seconds)
            labels = []
            for r in results:
                if r.t >= CALIBRATION_S:
                    labels.append(r.label)

            majority, share = majority_label(labels)

            expected = row["expected"]
            match = majority == expected

            results.append({
                "wav": row["wav"],
                "expected": expected,
                "predicted": majority,
                "share": share,
                "match": match,
                "distance_m": row.get("distance_m", ""),
                "noise": row.get("noise", ""),
                "notes": row.get("notes", ""),
            })

            status = "PASS" if match else "FAIL"
            print(f"  {row['wav']}: expected={expected}, got={majority} ({share*100:.0f}%) -> {status}")

    # Write markdown report
    with open(out_path, "w") as f:
        f.write("# Audio Evaluation Results\n\n")
        f.write(f"Manifest: {args.manifest}\n")
        f.write(f"Date: {__import__('datetime').datetime.now().isoformat()}\n\n")
        f.write("| WAV | Expected | Predicted | Share | Match | Distance | Noise |\n")
        f.write("|-----|----------|-----------|-------|-------|----------|-------|\n")
        for r in results:
            f.write(f"| {r['wav']} | {r['expected']} | {r['predicted']} | "
                    f"{r['share']*100:.0f}% | {'✓' if r['match'] else '✗'} | "
                    f"{r['distance_m']} | {r['noise']} |\n")

        # Per-class confusion
        f.write("\n## Per-Class Confusion\n\n")
        f.write("| Expected \\ Predicted | silence | speech | whisper | noise | calibrating |\n")
        f.write("|---|---|---|---|---|---|\n")
        classes = ["silence", "speech", "whisper", "noise", "calibrating"]
        for exp in classes:
            row = f"| {exp} "
            for pred in classes:
                count = sum(1 for r in results if r["expected"] == exp and r["predicted"] == pred)
                row += f"| {count} "
            row += "|\n"
            f.write(row)

        # Detection rate by distance
        f.write("\n## Detection Rate by Distance\n\n")
        f.write("| Distance (m) | Speech | Whisper | Noise | Silence |\n")
        f.write("|---|---|---|---|---|\n")
        for dist in sorted(set(r["distance_m"] for r in results if r["distance_m"])):
            subset = [r for r in results if r["distance_m"] == dist]
            if not subset:
                continue
            speech_rate = sum(1 for r in subset if r["predicted"] == "speech" and r["expected"] == "speech") / max(1, sum(1 for r in subset if r["expected"] == "speech"))
            whisper_rate = sum(1 for r in subset if r["predicted"] == "whisper" and r["expected"] == "whisper") / max(1, sum(1 for r in subset if r["expected"] == "whisper"))
            noise_fp = sum(1 for r in subset if r["predicted"] == "noise" and r["expected"] == "silence") / max(1, sum(1 for r in subset if r["expected"] == "silence"))
            silence_fn = sum(1 for r in subset if r["predicted"] == "silence" and r["expected"] in ("speech", "whisper")) / max(1, sum(1 for r in subset if r["expected"] in ("speech", "whisper")))
            f.write(f"| {dist} | {speech_rate:.1%} | {whisper_rate:.1%} | {noise_fp:.1%} | {silence_fn:.1%} |\n")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()