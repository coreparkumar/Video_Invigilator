"""
Edge Invigilator POC with Audio: video + audio gesture detection from a laptop webcam.

Single laptop, single camera, fully local (no network, no cloud, no continuous video
except a snapshot when a flagged event fires).

Run:   python gesture_poc_av.py
Keys:  c = calibrate neutral posture (sit normally, look at screen, then press)
       n = recalibrate noise floor (stay quiet, then press)
       m = mute/unmute audio
       s = toggle skeleton overlay
       t = toggle transcript box
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

# New UI imports
from ui.window import WindowManager, build_panel_state
from ui.findings import create_findings_recorder


def open_source(path):
    """Open video file source."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video file: {path}")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="preferred camera index")
    ap.add_argument("--audio-device", type=int, default=None, help="audio input device index")
    ap.add_argument("--video-file", type=str, default=None, help="video file for replay")
    ap.add_argument("--audio-file", type=str, default=None, help="audio file for replay")
    ap.add_argument("--out", default="evidence", help="folder for snapshots + log")
    ap.add_argument("--no-snapshots", action="store_true", help="disable snapshot saving")
    ap.add_argument("--no-audio", action="store_true", help="disable audio processing")

    # Dashboard options
    ap.add_argument("--dashboard", action="store_true", default=True, help="enable new dashboard (default)")
    ap.add_argument("--classic", action="store_true", help="use classic overlay instead of dashboard")
    ap.add_argument("--transcribe", action="store_true", help="enable local speech-to-text (requires --model)")
    ap.add_argument("--model", type=str, default=None, help="path to Vosk model folder")
    ap.add_argument("--log-transcript", action="store_true", help="log transcript to CSV (requires consent.txt)")

    args = ap.parse_args()

    # Validate transcription args
    if args.transcribe and not args.model:
        raise SystemExit("--transcribe requires --model <path to Vosk model folder>")
    if args.log_transcript and not os.path.exists("consent.txt"):
        print("WARNING: --log-transcript requires consent.txt next to the app. Transcript will not be saved.")

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

    # Window manager (dashboard)
    use_dashboard = not args.classic and args.dashboard
    window_manager = None
    findings_recorder = None
    transcript_enabled = args.transcribe
    show_transcript = False

    if use_dashboard:
        # Initialize window manager
        def on_download():
            if findings_recorder and findings_recorder.can_export():
                return findings_recorder.export_markdown(args.out)
            return ""

        window_manager = WindowManager(on_download=on_download)

        # Findings recorder
        findings_recorder = create_findings_recorder(time.time(), consent_path="consent.txt" if args.log_transcript else None)

        # Transcript state
        if args.transcribe:
            print("Transcription enabled (offline, local). Press 't' to toggle transcript box.")

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

            if use_dashboard:
                # Build panel state from session and audio engine
                session_view = session.get_view()
                audio_engine_state = None
                if audio_engine:
                    audio_engine_state = audio_engine.get_state()

                # Build panel state for dashboard
                panel_state = build_panel_state(
                    session_view=session_view,
                    audio_engine_state=audio_engine_state,
                    audio_events_count=len(audio_alerts) if audio_alerts else 0,
                    fused_alerts=audio_alerts,
                    last_audio_event=None,  # Could be enhanced
                    flags_count=session.flags_count if hasattr(session, 'flags_count') else 0,
                )

                # Update findings recorder
                if findings_recorder:
                    findings_recorder.update_video(panel_state, now)
                    findings_recorder.update_audio(panel_state, now)
                    for alert in audio_alerts:
                        findings_recorder.update_fused_alert(alert, now)

                # Render dashboard
                camera_frame = frame if use_dashboard else None
                canvas = window_manager.render(
                    panel_state=panel_state,
                    camera_frame=camera_frame,
                    landmarks=res.pose_landmarks.landmark if res and res.pose_landmarks else None,
                    connections=mp.solutions.pose.POSE_CONNECTIONS,
                    draw_spec=None,
                    spectrogram_heat=None,  # Could be enhanced with audio pipeline data
                    now=now,
                )

                # Show transcript box if enabled
                if transcript_enabled and show_transcript:
                    # Transcript box would be rendered in window_manager
                    pass

                cv2.imshow("Edge Invigilator (Dashboard)", canvas)

            else:
                # Classic overlay mode
                session_view = session.get_view()
                if audio_engine:
                    session_view["audio_calibrating"] = audio_engine.calibrating
                    session_view["audio_muted"] = audio_engine.muted
                    session_view["audio_floor_db"] = audio_engine._pipeline.floor.mean_db
                    session_view["audio_threshold_db"] = audio_engine._pipeline.floor.threshold_db

                frame = render_overlay(frame, session_view, fps, now)

                # Audio meter overlay
                if audio_engine and show_audio_meter:
                    h, w = frame.shape[:2]
                    floor_db = audio_engine._pipeline.floor.mean_db
                    thresh_db = audio_engine._pipeline.floor.threshold_db
                    level_db = audio_engine._pipeline.get_state().get("floor_mean_db", -60)
                    meter_x = w - 80
                    meter_y = 50
                    meter_h = 200
                    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + 30, meter_y + meter_h), (0, 0, 0), -1)
                    if floor_db is not None:
                        floor_y = int(meter_y + meter_h * (1 - (floor_db + 60) / 40))
                        cv2.line(frame, (meter_x, floor_y), (meter_x + 30, floor_y), (100, 100, 100), 1)
                    if thresh_db is not None:
                        thresh_y = int(meter_y + meter_h * (1 - (thresh_db + 60) / 40))
                        cv2.line(frame, (meter_x, thresh_y), (meter_x + 30, thresh_y), (0, 255, 255), 1)
                    if level_db is not None:
                        level_y = int(meter_y + meter_h * (1 - (level_db + 60) / 40))
                        level_y = max(meter_y, min(meter_y + meter_h, level_y))
                        cv2.line(frame, (meter_x, level_y), (meter_x + 30, level_y), (0, 255, 0), 2)
                    if audio_engine.calibrating:
                        cv2.putText(frame, "CALIBRATING...", (meter_x - 60, meter_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                    elif audio_engine.calibration_status.value != "ok":
                        msg = audio_engine.calibration_message
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
                    if audio_engine.clipping:
                        cv2.putText(frame, "CLIPPING!", (meter_x - 60, meter_y + meter_h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("Edge Invigilator POC (Video + Audio)", frame)

            # Handle keys
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
            elif key == ord("t"):
                show_transcript = not show_transcript
                if show_transcript:
                    print("Transcript box ON")
                else:
                    print("Transcript box OFF")

    finally:
        # Cleanup
        video_source.release()
        if audio_source:
            audio_source.close()
        if audio_engine:
            audio_engine.close()
        if findings_recorder:
            findings_recorder.finalize(time.time())
            if findings_recorder.can_export():
                findings_recorder.export_markdown(args.out)
        pose.close()
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()