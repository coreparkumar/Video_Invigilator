"""Live AV check wrapper: runs guided check with real camera and microphone.

Usage: python tools/av_check.py [--no-video] [--no-audio] [--device N] [--camera N] [--stages 1,3,5]
"""
import argparse
import sys
import time
import os
from pathlib import Path
import cv2
import numpy as np

from audio.engine import AudioEngine
from audio.camera import open_camera as open_audio_camera
from audio.capture import FakeAudioSource
from invigilator.camera import open_camera
from invigilator.pose import PoseEstimator
from invigilator.session import InvigilatorSession
from invigilator.overlay import render_overlay
from audio.avcheck import AVCheckRunner, default_stages, Stage, StageStatus
from audio.config import HOP_S


def get_video_active(session: InvigilatorSession) -> set:
    """Get active video cues from session."""
    return session.active if session.active else set()


def run_live_check(
    args,
    session: InvigilatorSession,
    audio_engine,
    pose,
    video_source,
    audio_source,
):
    """Run the live AV check with camera and microphone."""
    # Determine which stages to run
    if args.stages:
        stage_indices = [int(s.strip()) - 1 for s in args.stages.split(",")]
        stages = [default_stages()[i] for i in stage_indices if 0 <= i < len(default_stages())]
    else:
        stages = default_stages()

    print(f"Running {len(stages)} stages...")

    # Prepare clock
    class RealClock:
        def __call__(self):
            return time.monotonic()
        def advance(self, s):
            pass  # Real clock doesn't advance manually

    clock = lambda: time.monotonic()

    def get_video_active():
        return get_video_active(session)

    runner = AVCheckRunner(
        engine=audio_engine,
        get_video_active=get_video_active,
        clock=time.monotonic,
        sleep=time.sleep,
    )

    # Run stages
    results = []
    for stage in stages:
        print(f"\n{'='*60}")
        print(f"Stage {stage.name}: {stage.instruction}")
        print(f"Duration: {stage.seconds}s")
        print(f"Expected: {stage.expect}")

        if stage.name == "device_check":
            print(f"  Camera: {'OK' if video_source else 'FAIL'}")
            print(f"  Microphone: {'OK' if audio_engine else 'DISABLED'}")
            result = StageResult(
                name=stage.name,
                status=StageStatus.PASS if video_source and audio_engine else StageStatus.FAIL,
                measurements={},
                note="Device check"
            )
            results.append(result)
            continue

        if stage.name == "audio_calibration" and not audio_engine:
            print("  Skipped: audio disabled")
            result = StageResult(
                name=stage.name,
                status=StageStatus.SKIPPED,
                measurements={},
                note="Audio disabled"
            )
            results.append(result)
            continue

        if "video" in stage.name.lower() and not video_source:
            print("  Skipped: video disabled")
            result = StageResult(
                name=stage.name,
                status=StageStatus.SKIPPED,
                measurements={},
                note="Video disabled"
            )
            results.append(result)
            continue

        print(f"  Press ENTER to begin stage...")
        input()

        # Run stage
        start_time = time.monotonic()
        end_time = start_time + stage.seconds

        # For calibration stage, we need to track calibration
        if stage.name == "audio_calibration":
            print("  Stay quiet...")

        last_time = time.monotonic()
        video_cue_starts = {}
        video_cue_durations = {}
        alerts_logged = []

        while time.monotonic() < end_time:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            # Read video
            ok, frame = video_source.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            # Process video
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks.landmark if res.pose_landmarks else None

            if lm:
                session.update(lm, frame, now)

            # Get video active cues
            video_active = get_video_active(session)

            # Track video cue durations
            for cue in video_active:
                if cue not in video_cue_starts:
                    video_cue_starts[cue] = now
            for cue in list(video_cue_starts.keys()):
                if cue not in video_active:
                    dur = now - video_cue_starts[cue]
                    video_cue_durations[cue] = video_cue_durations.get(cue, 0.0) + dur
                    del video_cue_starts[cue]

            # Process audio
            if audio_engine:
                audio_alerts = audio_engine.step(now, video_active=video_active)
                for alert in audio_alerts:
                    alerts_logged.append(alert)

            # Render overlay
            session_view = session.get_view()
            if audio_engine:
                session_view["audio_calibrating"] = audio_engine.calibrating
                session_view["audio_muted"] = audio_engine.muted
                session_view["audio_floor_db"] = audio_engine._pipeline.floor.mean_db
                session_view["audio_threshold_db"] = audio_engine._pipeline.floor.threshold_db

            frame = render_overlay(frame, session_view, 30.0, now)

            # Draw stage info
            h, w = frame.shape[:2]
            remaining = max(0, int(end_time - time.monotonic()))
            cv2.putText(frame, f"Stage: {stage.name} ({remaining}s)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, stage.instruction, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Edge Invigilator AV Check", frame)
            cv2.waitKey(1)

        # Finalize video cue durations
        for cue, start in video_cue_starts.items():
            dur = time.monotonic() - start
            video_cue_durations[cue] = video_cue_durations.get(cue, 0.0) + dur

        # Collect measurements
        audio_events = []
        alerts = []
        if audio_engine:
            audio_events = list(audio_engine._session._history) if hasattr(audio_engine, '_session') else []
            # We need to track alerts during the stage

        measurements = {
            "video_cue_durations": video_cue_durations,
            "alerts": [],
            "calibration_ok": True,  # TODO
        }

        # Evaluate stage
        # TODO: proper evaluation

        print(f"  Stage {stage.name} completed")

    # Write report
    report_path = Path("docs/AV_CHECK_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# AV Check Report\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Device: Camera={video_source}, Audio={'Enabled' if audio_engine else 'Disabled'}\n\n")
        f.write("## Stages\n\n")
        f.write("| Stage | Status | Note |\n")
        f.write("|-------|--------|------|\n")
        for r in results:
            f.write(f"| {r.name} | {r.status.value} | {r.note} |\n")

        f.write("\n## Calibration\n\n")
        f.write(f"- Status: _\n")
        f.write(f"- Message: _\n")

        f.write("\n## Notes\n")
        f.write("- One run is not accuracy evidence. See AUDIO_EXTENSION.md for evaluation protocol.\n")
        f.write("- UNTESTED items marked as such.\n")

    print(f"\nReport written to {report_path}")


def main():
    ap = argparse.ArgumentParser(description="Live AV check with camera and microphone")
    ap.add_argument("--camera", type=int, default=0, help="Video camera index")
    ap.add_argument("--device", type=int, default=None, help="Audio input device index")
    ap.add_argument("--no-video", action="store_true", help="Disable video")
    ap.add_argument("--no-audio", action="store_true", help="Disable audio")
    ap.add_argument("--stages", type=str, help="Comma-separated stage numbers to run (1-9)")
    ap.add_argument("--out", default="evidence", help="Output folder")
    args = ap.parse_args()

    # Open video source
    video_source = None
    if not args.no_video:
        try:
            video_source = open_camera(args.camera)
            print(f"Opened camera index {args.camera}")
        except SystemExit as e:
            print(f"Warning: {e}")
            video_source = None

    # Open audio source
    audio_source = None
    if not args.no_audio:
        try:
            audio_source = open_audio_camera(args.device)
            print("Opened audio device")
        except RuntimeError as e:
            print(f"Warning: {e}")
            audio_source = None

    # Initialize modules
    session = InvigilatorSession(args.out, enable_snapshots=True)
    pose = PoseEstimator()
    audio_engine = None
    if audio_source:
        audio_engine = AudioEngine(audio_source, enabled=True)

    try:
        run_live_check(args, session, audio_engine, pose, video_source, audio_source)
    finally:
        if video_source:
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