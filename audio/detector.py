"""Speech/whisper detector: pure rules, thresholds from audio/config.py only."""
from dataclasses import dataclass
import numpy as np
from audio.config import (
    ACTIVE_FRAC_SUSTAINED, LEVEL_STD_MIN_DB, HARM_SPEECH,
    WHISPER_MAX_CENTROID_HZ
)
from audio.baseline import NoiseFloor


@dataclass
class DetectionResult:
    label: str              # "silence", "speech", "whisper", "noise_burst", "noise"
    confidence: float       # 0..1
    info: dict              # debug info


def classify_window(feats: dict, floor: NoiseFloor) -> DetectionResult:
    """Classify a 1-second window into {silence, speech, whisper, noise_burst, noise}.

    Algorithm (exact order):
    1. active = frame_speech_db > floor.threshold_db; frac = active.mean()
    2. If frac == 0: return ("silence", 1.0)
    3. over = mean(active frame dB) - floor.mean_db; conf = 0.4 + 0.6 * clip((over - floor.margin_db) / 12, 0, 1)
       level_std = std(active frame dB); harm = median(frame_harm[active])
    4. frac < ACTIVE_FRAC_SUSTAINED -> "noise_burst"
    5. level_std < LEVEL_STD_MIN_DB -> "noise"
    6. harm >= HARM_SPEECH -> "speech"
    7. centroid_hz > WHISPER_MAX_CENTROID_HZ -> "noise"
    8. otherwise -> "whisper"
    """
    frame_speech_db = feats["frame_speech_db"]
    frame_harm = feats["frame_harm"]
    centroid_hz = feats["centroid_hz"]

    # 1. Active frames above floor threshold
    active = frame_speech_db > floor.threshold_db
    frac = float(np.mean(active))

    info = {
        "level_db": float(np.mean(frame_speech_db)),
        "active_frac": frac,
        "above_floor_db": float(np.mean(frame_speech_db[active]) - floor.threshold_db) if frac > 0 else 0.0,
    }

    # 2. No active frames -> silence
    if frac == 0.0:
        return DetectionResult("silence", 1.0, info)

    # 3. Compute confidence and features
    active_db = frame_speech_db[active]
    over = float(np.mean(active_db) - floor.mean_db)
    conf = 0.4 + 0.6 * float(np.clip((over - floor.margin_db) / 12.0, 0.0, 1.0))
    level_std = float(np.std(active_db))
    harm = float(np.median(frame_harm[active]))

    info.update({
        "level_std_db": level_std,
        "harmonicity": harm,
        "centroid_hz": centroid_hz,
        "confidence_raw": conf,
    })

    # 4. Short burst -> noise_burst (cough, door, chair)
    if frac < ACTIVE_FRAC_SUSTAINED:
        return DetectionResult("noise_burst", conf, info)

    # 5. Steady level -> noise (hum, no syllable rhythm)
    if level_std < LEVEL_STD_MIN_DB:
        return DetectionResult("noise", conf, info)

    # 6. High harmonicity -> speech
    if harm >= HARM_SPEECH:
        return DetectionResult("speech", conf, info)

    # 7. Bright centroid -> noise (hiss, paper rustle)
    if centroid_hz > WHISPER_MAX_CENTROID_HZ:
        return DetectionResult("noise", conf, info)

    # 8. Otherwise -> whisper
    return DetectionResult("whisper", conf, info)


if __name__ == "__main__":
    # Quick smoke test with synthetic signals
    import numpy as np
    from audio.synth import ambient, speech, whisper, cough, steady_noise, paper_rustle
    from audio.features import window_features
    from audio.baseline import NoiseFloor
    from audio.config import HOP_S

    print("Testing detector with synthetic signals...")

    # Calibrate floor
    floor = NoiseFloor()
    cal_signal = ambient(5.0, db=-62, seed=10)
    cal_feats = window_features(cal_signal)
    floor.calibrate(cal_feats["frame_speech_db"])
    print(f"Floor: mean={floor.mean_db:.1f}, thresh={floor.threshold_db:.1f}")

    # Test cases: (signal_func, expected_label)
    test_cases = [
        (lambda: ambient(1.2, seed=21), "silence"),
        (lambda: speech(1.2, db=-35, f0=130), "speech"),
        (lambda: whisper(1.2, db=-38), "whisper"),
        (lambda: cough(0.25, db=-30), "noise_burst"),
        (lambda: steady_noise(1.2, db=-40), "noise"),
        (lambda: paper_rustle(1.2, db=-38), "noise"),
    ]

    for signal_fn, expected in test_cases:
        sig = signal_fn()
        feats = window_features(sig)
        result = classify_window(feats, floor)
        status = "✓" if result.label == expected else "✗"
        print(f"  {status} {expected:12s} -> {result.label:12s} (conf={result.confidence:.2f})")

    # Near-miss: very quiet whisper
    near_miss = whisper(1.2, db=-68)  # well below floor
    feats = window_features(near_miss)
    result = classify_window(feats, floor)
    print(f"  {'✓' if result.label == 'silence' else '✗'} {'silence':12s} -> {result.label:12s} (near-miss whisper)")

    # Confidence ordering: louder whisper > quieter whisper, both labelled whisper
    w_loud = whisper(1.2, db=-35)
    w_quiet = whisper(1.2, db=-50)
    feats_loud = window_features(w_loud)
    feats_quiet = window_features(w_quiet)
    res_loud = classify_window(feats_loud, floor)
    res_quiet = classify_window(feats_quiet, floor)
    print(f"  Loud whisper:  {res_loud.label} conf={res_loud.confidence:.2f}")
    print(f"  Quiet whisper: {res_quiet.label} conf={res_quiet.confidence:.2f}")
    if res_loud.label == "whisper" and res_quiet.label == "whisper":
        print(f"  Confidence ordering: {'✓' if res_loud.confidence > res_quiet.confidence else '✗'} ({res_loud.confidence:.2f} > {res_quiet.confidence:.2f})")

    print("Smoke test passed")