"""Evaluation harness: guided accuracy test and false-positive soak test."""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np

from invigilator.camera import open_source
from invigilator.pose import PoseEstimator
from invigilator.session import Session
from invigilator.evidence import EvidenceLog
from invigilator.overlay import render
from invigilator.config import GESTURES


GESTURE_ORDER = [
    ("hand_raised", "Raise your hand above your head"),
    ("hand_to_ear", "Place hand at your ear (phone/earpiece gesture)"),
    ("hand_to_face", "Cover your mouth/nose with hand"),
    ("look_left", "Look to your LEFT (away from screen)"),
    ("look_right", "Look to your RIGHT (away from screen)"),
    ("look_down", "Look DOWN at your desk/notes"),
    ("leaning", "Lean sideways in your chair"),
]


def eval_gestures(args):
    """Guided accuracy evaluation for each gesture."""
    print("=" * 60)
    print("EDGE INVIGILATOR - GESTURE ACCURACY EVALUATION")
    print("=" * 60)
    print()
    print("You will be prompted to perform each gesture 10 times.")
    print("Hold each gesture for 3 seconds (longer than threshold).")
    print("Press ENTER when ready for each attempt.")
    print()

    # Open camera
    source = open_source(None, preferred_cam=args.camera)
    pose = PoseEstimator()
    session = Session()
    evidence = EvidenceLog(args.out)

    results = {g: {"hits": 0, "total": 0} for g, _ in GESTURE_ORDER}

    try:
        for gesture_key, prompt in GESTURE_ORDER:
            label, need = GESTURES[gesture_key]
            print(f"\n{'='*60}")
            print(f"GESTURE: {label} (hold {need:.1f}s)")
            print(f"PROMPT: {prompt}")
            print(f"{'='*60}")

            for attempt in range(1, 11):
                input(f"\nAttempt {attempt}/10 - Press ENTER when ready...")
                print("Perform gesture NOW and hold for 3 seconds...")

                # Run for 3 seconds
                start = time.time()
                detected = False

                while time.time() - start < 3.0:
                    ok, frame = source.read()
                    if not ok:
                        break

                    frame = cv2.flip(frame, 1)
                    now = time.time()

                    landmarks, _ = pose.process(frame)
                    flags = session.update(landmarks, now)

                    for flag in flags:
                        if flag.gesture_key == gesture_key:
                            detected = True
                            print(f"  DETECTED! (held {flag.held_seconds:.1f}s)")

                    view = session.get_view()
                    frame = render(frame, view, 30.0, now)
                    cv2.imshow("Edge Invigilator POC", frame)
                    cv2.waitKey(1)

                if detected:
                    results[gesture_key]["hits"] += 1
                    print(f"  Result: HIT ({results[gesture_key]['hits']}/{attempt})")
                else:
                    print(f"  Result: MISS ({results[gesture_key]['hits']}/{attempt})")

                results[gesture_key]["total"] += 1

                # Brief pause between attempts
                time.sleep(0.5)

    finally:
        source.release()
        pose.close()
        evidence.close()
        cv2.destroyAllWindows()

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Gesture':<25} {'Hits/10':<10} {'Pass (>=8)':<12}")
    print("-" * 60)

    all_passed = True
    for gesture_key, prompt in GESTURE_ORDER:
        label, _ = GESTURES[gesture_key]
        hits = results[gesture_key]["hits"]
        passed = "PASS" if hits >= 8 else "FAIL"
        if hits < 8:
            all_passed = False
        print(f"{label:<25} {hits}/10       {passed}")

    print("-" * 60)
    print(f"Overall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    # Write markdown results
    md_path = Path(args.out) / "EVAL.md"
    with open(md_path, "w") as f:
        f.write("# Gesture Accuracy Evaluation Results\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Gesture | Hits/10 | Pass |\n")
        f.write("|---------|---------|------|\n")
        for gesture_key, prompt in GESTURE_ORDER:
            label, _ = GESTURES[gesture_key]
            hits = results[gesture_key]["hits"]
            passed = "✓" if hits >= 8 else "✗"
            f.write(f"| {label} | {hits}/10 | {passed} |\n")
        f.write(f"\n**Overall:** {'PASS' if all_passed else 'FAIL'}\n")

    print(f"\nResults written to {md_path}")
    return all_passed


def soak_test(args):
    """False-positive soak test: sit and type normally for N minutes."""
    print("=" * 60)
    print("EDGE INVIGILATOR - FALSE POSITIVE SOAK TEST")
    print("=" * 60)
    print()
    print(f"Duration: {args.minutes} minutes")
    print("Sit and type/work normally. Do NOT perform any gestures.")
    print("Press ENTER to start...")
    input()

    source = open_source(None, preferred_cam=args.camera)
    pose = PoseEstimator()
    session = Session()
    evidence = EvidenceLog(args.out)

    start_time = time.time()
    end_time = start_time + args.minutes * 60
    frame_count = 0
    flags_detected = []

    print(f"Starting soak test at {time.strftime('%H:%M:%S')}")
    print(f"Will end at {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    print()

    try:
        while time.time() < end_time:
            ok, frame = source.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            now = time.time()

            landmarks, _ = pose.process(frame)
            flags = session.update(landmarks, now)

            for flag in flags:
                flags_detected.append((flag.gesture_key, flag.label, flag.held_seconds))
                print(f"  FLAG: {flag.label} (held {flag.held_seconds:.1f}s)")

            view = session.get_view()
            frame = render(frame, view, 30.0, now)
            cv2.imshow("Edge Invigilator POC (Soak Test)", frame)
            cv2.waitKey(1)

            frame_count += 1

            # Progress update every 30 seconds
            elapsed = time.time() - start_time
            if frame_count % (30 * 30) == 0:  # ~every 30 seconds at 30fps
                fps = frame_count / elapsed
                remaining = int(end_time - time.time())
                print(f"  Progress: {int(elapsed/60)}/{args.minutes} min, "
                      f"FPS: {fps:.1f}, Flags so far: {len(flags_detected)}")

    finally:
        source.release()
        pose.close()
        evidence.close()
        cv2.destroyAllWindows()

    # Summary
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("SOAK TEST SUMMARY")
    print("=" * 60)
    print(f"Duration: {elapsed/60:.1f} minutes")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Flags detected: {len(flags_detected)}")

    if flags_detected:
        print("\nFlags:")
        for g, label, held in flags_detected:
            print(f"  - {label} (held {held:.1f}s)")
        print("\nRESULT: FAIL - False positives detected")
    else:
        print("\nRESULT: PASS - Zero false positives")

    # Write markdown
    md_path = Path(args.out) / "SOAK.md"
    with open(md_path, "w") as f:
        f.write("# False Positive Soak Test Results\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {elapsed/60:.1f} minutes\n")
        f.write(f"Frames: {frame_count}\n")
        f.write(f"Average FPS: {fps:.1f}\n")
        f.write(f"Flags detected: {len(flags_detected)}\n\n")
        if flags_detected:
            f.write("## Flags (FALSE POSITIVES)\n\n")
            for g, label, held in flags_detected:
                f.write(f"- {label} (held {held:.1f}s)\n")
            f.write("\n**RESULT: FAIL**\n")
        else:
            f.write("\n**RESULT: PASS** - Zero false positives\n")

    print(f"\nResults written to {md_path}")
    return len(flags_detected) == 0


def main():
    ap = argparse.ArgumentParser(description="Edge Invigilator Evaluation Tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # eval_gestures
    ap_eval = sub.add_parser("eval", help="Guided gesture accuracy test")
    ap_eval.add_argument("--camera", type=int, default=0, help="Camera index")
    ap_eval.add_argument("--out", default="evidence", help="Output folder")

    # soak_test
    ap_soak = sub.add_parser("soak", help="False-positive soak test")
    ap_soak.add_argument("--camera", type=int, default=0, help="Camera index")
    ap_soak.add_argument("--out", default="evidence", help="Output folder")
    ap_soak.add_argument("--minutes", type=int, default=2, help="Test duration in minutes")

    args = ap.parse_args()

    if args.cmd == "eval":
        eval_gestures(args)
    elif args.cmd == "soak":
        soak_test(args)


if __name__ == "__main__":
    main()