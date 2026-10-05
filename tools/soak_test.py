"""Soak test: false-positive detection during normal use.

Runs for N minutes while user sits and types normally.
Reports flags, FPS, and memory usage.
"""
# This module is imported by eval_gestures.py which contains both eval and soak.
# Keeping this as a separate entry point for backward compatibility.

from tools.eval_gestures import soak_test
import argparse


def main():
    ap = argparse.ArgumentParser(description="False-positive soak test")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--out", default="evidence", help="Output folder")
    ap.add_argument("--minutes", type=int, default=2, help="Test duration in minutes")
    args = ap.parse_args()

    soak_test(args)


if __name__ == "__main__":
    main()