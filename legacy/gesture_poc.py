"""
Edge Invigilator POC: body-gesture detection from a laptop webcam.

Single laptop, single camera, fully local (no network, no cloud, no video saved
except a snapshot when a flagged event fires).

Run:   python gesture_poc.py
Keys:  c = calibrate neutral posture (sit normally, look at screen, then press)
       s = toggle skeleton overlay
       q = quit
"""
import argparse
import csv
import math
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
import mediapipe as mp

# ==========================================================================
# CONFIG: everything you may want to tune lives here
# ==========================================================================
# gesture -> (label, seconds it must persist before it is FLAGGED)
GESTURES = {
    "hand_raised":   ("Hand raised",                   1.0),
    "hand_to_ear":   ("Hand at ear (phone/earpiece?)", 1.5),
    "hand_to_face":  ("Hand covering face/mouth",      2.0),
    "look_left":     ("Looking left",                  2.5),
    "look_right":    ("Looking right",                 2.5),
    "look_up":       ("Looking up",                    2.5),
    "look_down":     ("Looking down",                  4.0),
    "leaning":       ("Leaning sideways",              2.5),
    "out_of_frame":  ("Out of frame / left seat",      3.0),
}

# Geometric limits. "sh_w" = shoulder width, so values scale with camera distance.
THRESHOLDS = {
    "hand_raised_above_nose": 0.15,  # wrist this far above nose (x sh_w)
    "hand_to_face_dist":      0.25,  # wrist-to-nose distance (x sh_w)
    "hand_to_ear_dist":       0.30,  # wrist-to-ear distance (x sh_w)
    "yaw_shift":              0.45,  # nose offset from ear midpoint vs baseline (x ear width)
    "neck_ratio":             0.55,  # looking down if neck < baseline * this
    "neck_ratio_up":          1.25,  # looking up if neck > baseline * this (UNTUNED)
    "tilt_deg":               14.0,  # shoulder-line tilt change vs baseline (degrees)
}

DEFAULT_BASELINE = {"yaw": 0.0, "neck": 0.65, "tilt": 0.0}   # used before calibration
COOLDOWN_S = 5.0          # min gap between repeat flags of the same gesture
VIS_MIN = 0.5             # landmark visibility threshold
VIS_MIN_PARTIAL = 0.3     # pose found but key landmarks below this -> PARTIAL (UNTUNED)
ABSENT_GRACE_S = 0.5      # no-person time before "out of frame" starts counting

# Dynamic baseline: slowly follows the user's neutral posture (chair shifts etc.)
BASELINE_TAU_S = 20.0     # time constant; larger = slower adaptation
BASELINE_GATE = 0.5       # only adapt while within this fraction of every threshold,
                          # so sustained near-threshold behaviour is not "learned away"
CAMERA_SCAN = range(0, 4)

# Pose landmark indices (MediaPipe Pose)
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH = 11, 12
L_WR, R_WR = 15, 16


# ==========================================================================
# Inference logic (pure functions, no UI / state)
# ==========================================================================
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def measures(lm):
    """Scale-invariant posture measurements from pose landmarks."""
    sh_w = max(dist(lm[L_SH], lm[R_SH]), 1e-6)
    mid_sh_y = (lm[L_SH].y + lm[R_SH].y) / 2
    ear_mid_x = (lm[L_EAR].x + lm[R_EAR].x) / 2
    ear_w = max(abs(lm[L_EAR].x - lm[R_EAR].x), 1e-6)
    return {
        "sh_w": sh_w,
        # >0: nose shifted toward image-right of ear midpoint (after mirror flip)
        "yaw": (lm[NOSE].x - ear_mid_x) / ear_w,
        # neck-length proxy: nose-to-shoulder-line vertical gap in shoulder widths
        "neck": (mid_sh_y - lm[NOSE].y) / sh_w,
        # shoulder line tilt in degrees
        "tilt": math.degrees(math.atan2(lm[R_SH].y - lm[L_SH].y,
                                        abs(lm[R_SH].x - lm[L_SH].x))),
    }


def classify(lm, base=None, th=THRESHOLDS):
    """Return the set of gesture keys currently active for one frame."""
    active = set()
    m = measures(lm)
    b = base or DEFAULT_BASELINE
    nose, sh_w = lm[NOSE], m["sh_w"]
    ears = [lm[L_EAR], lm[R_EAR]]

    for wr in (lm[L_WR], lm[R_WR]):
        if wr.visibility < VIS_MIN:
            continue
        above_nose = wr.y < nose.y - th["hand_raised_above_nose"] * sh_w
        if above_nose:
            active.add("hand_raised")
        d_ear = min(dist(wr, e) for e in ears)
        if dist(wr, nose) < th["hand_to_face_dist"] * sh_w:
            active.add("hand_to_face")
        elif d_ear < th["hand_to_ear_dist"] * sh_w and not above_nose:
            active.add("hand_to_ear")

    dyaw = m["yaw"] - b["yaw"]
    if dyaw > th["yaw_shift"]:
        active.add("look_right")
    elif dyaw < -th["yaw_shift"]:
        active.add("look_left")
    if m["neck"] < b["neck"] * th["neck_ratio"]:
        active.add("look_down")
    if m["neck"] > b["neck"] * th["neck_ratio_up"]:
        active.add("look_up")
    if abs(m["tilt"] - b["tilt"]) > th["tilt_deg"]:
        active.add("leaning")
    return active


def near_neutral(m, base, th=THRESHOLDS, gate=BASELINE_GATE):
    """True if posture is well inside every limit (safe to adapt baseline)."""
    return (abs(m["yaw"] - base["yaw"]) < gate * th["yaw_shift"]
            and m["neck"] > base["neck"] * (1 - gate * (1 - th["neck_ratio"]))
            and abs(m["tilt"] - base["tilt"]) < gate * th["tilt_deg"])


# ==========================================================================
# State management
# ==========================================================================
class InvigilatorSession:
    """Holds calibration, hold-timers, cooldowns and the evidence log."""

    def __init__(self, out_dir="evidence"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "events.csv")
        is_new = not os.path.exists(log_path)
        self._log_f = open(log_path, "a", newline="")
        self._log = csv.writer(self._log_f)
        if is_new:
            self._log.writerow(["timestamp", "gesture", "label", "held_seconds", "snapshot"])

        self.base = None                     # calibrated baseline (None = not calibrated)
        self.active = set()
        self.since = {}                      # gesture -> time it became active
        self.last_flag = defaultdict(float)
        self.banners = []                    # (expire_time, text)
        self._absent_since = None
        self._last_t = None
        self.frame_state = "OUT"             # IN, PARTIAL, OUT

    # -- calibration -------------------------------------------------------
    def calibrate(self, lm):
        self.base = {k: v for k, v in measures(lm).items() if k != "sh_w"}
        return self.base

    def _adapt_baseline(self, m, dt):
        alpha = 1 - math.exp(-dt / BASELINE_TAU_S)
        for k in self.base:
            self.base[k] += alpha * (m[k] - self.base[k])

    # -- per-frame update --------------------------------------------------
    def update(self, lm, frame, now):
        """lm: pose landmarks or None. Returns list of newly flagged labels."""
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        active = set()

        if lm is not None:
            self._absent_since = None
            # Check visibility of key landmarks (NOSE, L_SH, R_SH)
            key_vis = [lm[i].visibility for i in (NOSE, L_SH, R_SH)]
            min_vis = min(key_vis)

            if min_vis > VIS_MIN:
                # All key landmarks clearly visible -> IN
                self.frame_state = "IN"
                active = classify(lm, self.base)
                if self.base and not active:
                    m = measures(lm)
                    if near_neutral(m, self.base):
                        self._adapt_baseline(m, dt)
            elif min_vis > VIS_MIN_PARTIAL:
                # Pose found but some key landmarks have low visibility -> PARTIAL
                self.frame_state = "PARTIAL"
                # No classification, no baseline adaptation
            else:
                # Key landmarks too unreliable -> treat as no reliable pose
                self.frame_state = "PARTIAL"
        else:
            # No person detected
            self.frame_state = "OUT"
            self._absent_since = self._absent_since or now
            if now - self._absent_since > ABSENT_GRACE_S:
                active.add("out_of_frame")

        self.active = active
        for g in list(self.since):
            if g not in active:
                del self.since[g]

        new_flags = []
        for g in active:
            self.since.setdefault(g, now)
            label, need = GESTURES[g]
            held = now - self.since[g]
            if held >= need and now - self.last_flag[g] >= COOLDOWN_S:
                self.last_flag[g] = now
                self._record(g, label, held, frame)
                self.banners.append((now + 4, f"FLAG: {label}"))
                new_flags.append(label)
        self.banners = [(t, s) for t, s in self.banners if t > now]
        return new_flags

    def _record(self, gesture, label, held, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = os.path.join(self.out_dir, f"{ts}_{gesture}.jpg")
        cv2.imwrite(snap, frame)
        self._log.writerow([datetime.now().isoformat(timespec="seconds"),
                            gesture, label, f"{held:.1f}", snap])
        self._log_f.flush()
        print(f"[{ts}] FLAG {label} (held {held:.1f}s)")

    def close(self):
        self._log_f.close()


# ==========================================================================
# UI rendering
# ==========================================================================
def render_overlay(frame, session, fps, now):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (330, 30 + 24 * len(GESTURES)), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)
    for i, (g, (label, need)) in enumerate(GESTURES.items()):
        on = g in session.active
        held = now - session.since[g] if on else 0
        color = ((0, 0, 255) if on and held >= need
                 else (0, 200, 255) if on else (140, 140, 140))
        txt = f"{label}  {held:.1f}/{need:.1f}s" if on else label
        cv2.putText(frame, txt, (10, 28 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for j, (_, msg) in enumerate(session.banners[-3:]):
        cv2.putText(frame, msg, (w // 2 - 160, 40 + 34 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    status = "calibrated (adaptive)" if session.base else "NOT calibrated (press c)"
    cv2.putText(frame, f"{fps:.0f} FPS | {status} | c:calibrate s:skeleton q:quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ==========================================================================
# Camera
# ==========================================================================
def open_camera(preferred):
    """Try the requested index first, then auto-scan the others."""
    order = [preferred] + [i for i in CAMERA_SCAN if i != preferred]
    for idx in order:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened() and cap.read()[0]:
            print(f"Using camera index {idx}")
            return cap
        cap.release()
    raise SystemExit(
        f"No camera found at indices {list(CAMERA_SCAN)}. Check that it is not in use "
        "by another app and that your OS allows the terminal/IDE to access the camera "
        "(see README: Troubleshooting).")


# ==========================================================================
# Main loop
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="preferred index; others auto-scanned")
    ap.add_argument("--out", default="evidence", help="folder for snapshots + log")
    args = ap.parse_args()

    cap = open_camera(args.camera)
    session = InvigilatorSession(args.out)
    pose = mp.solutions.pose.Pose(model_complexity=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    drawer, styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles
    show_skel, fps, fps_t = True, 0.0, time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)   # mirror so left/right feel natural
            now = time.time()

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks.landmark if res.pose_landmarks else None
            if res.pose_landmarks and show_skel:
                drawer.draw_landmarks(
                    frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=styles.get_default_pose_landmarks_style())

            session.update(lm, frame, now)

            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
            fps_t = now
            cv2.imshow("Edge Invigilator POC", render_overlay(frame, session, fps, now))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                show_skel = not show_skel
            elif key == ord("c") and lm is not None:
                b = session.calibrate(lm)
                print("Calibrated:", {k: round(v, 3) for k, v in b.items()})
    finally:
        cap.release()
        pose.close()
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
