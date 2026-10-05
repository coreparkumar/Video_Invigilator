"""Main application: wires all modules into the runnable Edge Invigilator POC."""

import argparse
import sys
from typing import Optional, Any, Callable, List

import cv2
import numpy as np

from invigilator.camera import open_source, FrameSource
from invigilator.pose import PoseEstimator
from invigilator.session import Session
from invigilator.evidence import EvidenceLog
from invigilator.overlay import render, draw_skeleton
from invigilator.config import GESTURES


def run_loop(
    source: FrameSource,
    pose: PoseEstimator,
    session: Session,
    evidence: EvidenceLog,
    render_fn: Callable,
    keys: Optional[List[int]] = None,
) -> None:
    """Main processing loop.

    Args:
        source: FrameSource (camera, video file, or fake)
        pose: PoseEstimator instance
        session: Session instance
        evidence: EvidenceLog instance
        render_fn: Function to render overlay
        keys: Optional list of key codes to simulate (for testing)
    """
    show_skeleton = True
    fps = 0.0
    fps_t = 0.0
    key_idx = 0

    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break

            # Mirror flip for natural left/right
            frame = cv2.flip(frame, 1)
            now = cv2.getTickCount() / cv2.getTickFrequency()

            # Process pose
            landmarks, pose_result = pose.process(frame)
            if landmarks and show_skeleton:
                draw_skeleton(frame, landmarks, pose._connections)

            # Update session
            flags = session.update(landmarks, now)
            for flag in flags:
                evidence.record(flag.gesture_key, flag.label, flag.held_seconds, frame)

            # Calculate FPS
            if fps_t > 0:
                fps = 0.9 * fps + 0.1 / (now - fps_t)
            fps_t = now

            # Render overlay
            view = session.get_view()
            frame = render_fn(frame, view, fps, now)

            cv2.imshow("Edge Invigilator POC", frame)

            # Handle keys
            if keys is not None and key_idx < len(keys):
                key = keys[key_idx]
                key_idx += 1
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("s"):
                show_skeleton = not show_skeleton
            elif key == ord("c") and landmarks is not None:
                base = session.calibrate(landmarks)
                print(f"Calibrated: {{'yaw': {base['yaw']:.3f}, 'neck': {base['neck']:.3f}, 'tilt': {base['tilt']:.3f}}}")

    finally:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Edge Invigilator POC")
    ap.add_argument("--camera", type=int, default=0, help="Preferred camera index")
    ap.add_argument("--video", type=str, help="Video file path for replay")
    ap.add_argument("--out", default="evidence", help="Output folder for snapshots + log")
    ap.add_argument("--no-snapshots", action="store_true", help="Disable snapshot saving")
    args = ap.parse_args()

    # Open camera or video source
    if args.video:
        print(f"Opening video file: {args.video}")
        source = open_source(args.video)
    else:
        print(f"Opening camera (preferred index: {args.camera})...")
        source = open_source(None, preferred_cam=args.camera)

    # Initialize modules
    pose = PoseEstimator()
    session = Session()
    evidence = EvidenceLog(args.out) if not args.no_snapshots else None

    try:
        # Wrap evidence.record for no-snapshots mode
        if args.no_snapshots:
            class NullEvidence:
                def record(self, *args, **kwargs):
                    pass
                def close(self):
                    pass
            evidence = NullEvidence()

        run_loop(source, pose, session, evidence, render)

    finally:
        source.release()
        pose.close()
        evidence.close()


if __name__ == "__main__":
    main()