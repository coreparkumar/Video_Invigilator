"""
Edge Invigilator POC with Audio: video + audio gesture detection from a laptop webcam.

Single laptop, single camera, fully local (no network, no cloud, no continuous video
except a snapshot when a flagged event fires).

Run:   python gesture_poc_av.py
Keys:  c = calibrate neutral posture (sit normally, look at screen, then press)
       n = recalibrate noise floor (stay quiet, then press)
       m = mute/unmute audio
       s = toggle skeleton overlay
       q = quit
"""
import argparse
import os
import time

import cv2
import mediapipe as mp

from invigilator.config import (
    GESTURES, THRESHOLDS, DEFAULT_BASELINE, COOLDOWN_S, VIS_MIN,
    VIS_MIN_PARTIAL, ABSENT_GRACE_S, BASELINE_TAU_S, BASELINE_GATE, CAMERA_SCAN,
    NOSE, L_EAR, R_EAR, L_SH, R_SH, L_WR, R_WR,
)
from invigilator.session import InvigilatorSession
from invigilator.evidence import EvidenceLog
from invigilator.overlay import render as render_overlay
from invigilator.camera import open_camera
from invigilator.pose import PoseEstimator
from invigilator.features import measures, dist
from invigilator.classifier import classify

# Audio imports
from audio.engine import AudioEngine
from audio.camera import open_camera as open_audio_camera
from audio.config import HOP_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="preferred camera index")
    ap.add_argument("--audio-device", type=int, default=None, help="audio input device index")
    ap.add_argument("--video-file", type=str, default=None, help="video file for replay")
    ap.add_argument("--audio-file", type=str, default=None, help="audio file for replay")
    ap.add_argument("--out", default="evidence", help="folder for snapshots + log")
    ap.add_argument("--no-snapshots", action="store_true", help="disable snapshot saving")
    ap.add_argument("--no-audio", action="store_true", help="disable audio processing")
    args = ap.parse_args()

    # Open video source
    if args.video_file:
        print(f"Opening video file: {args.video_file}")
        video_source = open_source(args.video_file)
    else:
        video_source = open_camera(args.camera)

    # Open audio source
    audio_source = None
    if not args.no_audio:
        if args.audio_file:
            print(f"Opening audio file: {args.audio_file}")
            from audio.capture import WavSource
            audio_source = WavSource(args.audio_file)
        else:
            print("Opening audio device...")
            audio_source = open_audio_camera(args.audio_device)

    # Initialize modules
    session = InvigilatorSession(args.out, enable_snapshots=not args.no_snapshots)
    pose = PoseEstimator()
    drawer, styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles
    show_skel = True

    # Audio engine
    audio_engine = None
    if audio_source:
        audio_engine = AudioEngine(audio_source, enabled=True)

    show_audio_meter = True
    fps, fps_t = 0.0, time.time()

    try:
        while True:
            # Read video frame
            ok, frame = video_source.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror
            now = time.time()

            # Process video pose
            lm, res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res and res.pose_landmarks and show_skel:
                drawer.draw_landmarks(
                    frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=styles.get_default_pose_landmarks_style())

            # Update video session
            session.update(lm, frame, now)

            # Process audio
            audio_alerts = []
            if audio_engine:
                video_active = session.active if lm else set()
                audio_alerts = audio_engine.step(now, video_active=video_active)

                # Log audio alerts to evidence
                for alert in audio_alerts:
                    session._record_audio_alert(alert, frame)

            # Calculate FPS
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
            fps_t = now

            # Render overlay
            session_view = session.get_view()
            # Add audio info to session view for overlay
            if audio_engine:
                session_view["audio_calibrating"] = audio_engine.calibrating
                session_view["audio_muted"] = audio_engine.muted
                session_view["audio_floor_db"] = audio_engine._pipeline.floor.mean_db
                session_view["audio_threshold_db"] = audio_engine._pipeline.floor.threshold_db

            frame = render_overlay(frame, session_view, fps, now)

            # Audio meter overlay (always show if audio engine exists and show_audio_meter is True)
            if audio_engine and show_audio_meter:
                h, w = frame.shape[:2]
                floor_db = audio_engine._pipeline.floor.mean_db
                thresh_db = audio_engine._pipeline.floor.threshold_db
                level_db = audio_engine._pipeline.get_state().get("floor_mean_db", -60)
                # Draw meter on right side
                meter_x = w - 80
                meter_y = 50
                meter_h = 200
                # Background
                cv2.rectangle(frame, (meter_x, meter_y), (meter_x + 30, meter_y + meter_h), (0, 0, 0), -1)
                # Floor line
                if floor_db is not None:
                    floor_y = int(meter_y + meter_h * (1 - (floor_db + 60) / 40))
                    cv2.line(frame, (meter_x, floor_y), (meter_x + 30, floor_y), (100, 100, 100), 1)
                # Threshold line
                if thresh_db is not None:
                    thresh_y = int(meter_y + meter_h * (1 - (thresh_db + 60) / 40))
                    cv2.line(frame, (meter_x, thresh_y), (meter_x + 30, thresh_y), (0, 255, 255), 1)
                # Current level
                if level_db is not None:
                    level_y = int(meter_y + meter_h * (1 - (level_db + 60) / 40))
                    level_y = max(meter_y, min(meter_y + meter_h, level_y))
                    cv2.line(frame, (meter_x, level_y), (meter_x + 30, level_y), (0, 255, 0), 2)

                # Calibration status message
                if audio_engine.calibrating:
                    cv2.putText(frame, "CALIBRATING...", (meter_x - 60, meter_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                elif audio_engine.calibration_status.value != "ok":
                    # Show calibration error message
                    msg = audio_engine.calibration_message
                    # Wrap long messages
                    lines = []
                    words = msg.split()
                    line = ""
                    for w in words:
                        if len(line + w) > 35:
                            lines.append(line)
                            line = w + " "
                        else:
                            line += w + " "
                    if line:
                        lines.append(line)
                    for i, line in enumerate(lines):
                        cv2.putText(frame, line, (meter_x - 60, meter_y - 10 - i * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Clipping indicator
                if audio_engine.clipping:
                    cv2.putText(frame, "CLIPPING!", (meter_x - 60, meter_y + meter_h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Edge Invigilator POC (Video + Audio)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                show_skel = not show_skel
            elif key == ord("c") and lm is not None:
                b = session.calibrate(lm)
                print("Calibrated:", {k: round(v, 3) for k, v in b.items()})
            elif key == ord("n") and audio_engine:
                audio_engine.recalibrate()
                print("Audio recalibration started")
            elif key == ord("m") and audio_engine:
                audio_engine.toggle_mute()
                print("Audio muted" if audio_engine.muted else "Audio unmuted")

    finally:
        video_source.release()
        if audio_source:
            audio_source.close()
        if audio_engine:
            audio_engine.close()
        pose.close()
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()