"""Live audio meter: real-time microphone monitoring with noise floor calibration.

Usage: python tools/audio_meter.py [--device N] [--list]
"""
import argparse
import sys
import time

from audio.capture import list_input_devices, open_camera as open_audio_camera
from audio.pipeline import AudioPipeline
from audio.config import SAMPLE_RATE, HOP_S


def main():
    ap = argparse.ArgumentParser(description="Live audio meter with noise floor calibration")
    ap.add_argument("--device", type=int, default=None, help="Audio input device index")
    ap.add_argument("--list", action="store_true", help="List available input devices and exit")
    args = ap.parse_args()

    if args.list:
        print("Available audio input devices:")
        try:
            devices = list_input_devices()
            for idx, name, chans in devices:
                print(f"  [{idx}] {name} ({chans} ch)")
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # Open audio source
    try:
        source = open_audio_camera(args.device)
        print(f"Opened audio device")
    except RuntimeError as e:
        print(f"Error opening audio device: {e}")
        print("Check that the device is not in use by another application.")
        print("See README: Troubleshooting for OS-specific camera/mic permissions.")
        sys.exit(1)

    # Create pipeline
    pipe = AudioPipeline(t0=0.0, auto_calibrate=True)
    print("Starting audio meter... Press Ctrl+C to stop")
    print("Calibration: stay quiet for 5 seconds...")

    try:
        frame_count = 0
        while True:
            samples = source.read()
            if len(samples) == 0:
                time.sleep(0.01)
                continue

            results = pipe.push(samples)
            for r in results:
                if r.label == "calibrating":
                    print(f"  CALIBRATING... level={r.level_db:.1f}dB", end="\r")
                else:
                    status = r.label.upper()
                    bar_len = int(max(0, min(40, (r.level_db + 60) / 60 * 40)))
                    bar = "#" * bar_len + "." * (40 - bar_len)
                    floor = pipe.floor.mean_db if pipe.floor.calibrated else "N/A"
                    thresh = pipe.floor.threshold_db if pipe.floor.calibrated else "N/A"
                    print(f"  {status:12s} |{bar}| {r.level_db:6.1f}dB  floor={floor} thresh={thresh}", end="\r")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        source.close()
        print("\nAudio meter stopped.")


if __name__ == "__main__":
    main()