"""All audio tunables in one place. Values are UNTUNED starting points: calibrate
on your own room and microphone (see AUDIO_EXTENSION.md)."""

# ---- capture / framing -----------------------------------------------------
SAMPLE_RATE = 16000
FRAME = 512              # 32 ms analysis frame
WINDOW_S = 1.0           # classification window
HOP_S = 0.5              # a new decision every 0.5 s
SPEECH_BAND = (300, 4000)
FLATNESS_BAND = (100, 8000)
PITCH_MIN_HZ, PITCH_MAX_HZ = 80, 400

# ---- noise floor -----------------------------------------------------------
CALIBRATION_S = 5.0      # stay quiet while this runs
MIN_MARGIN_DB = 8.0      # speech-band level must exceed floor by at least this
MARGIN_STD_MULT = 4.0    # ...or this many floor standard deviations
FLOOR_TAU_S = 60.0       # slow adaptation time constant
FLOOR_GATE = 0.5         # adapt only within this fraction of the margin

# ---- window classifier -----------------------------------------------------
ACTIVE_FRAC_SUSTAINED = 0.4   # below this share of active frames: short burst
LEVEL_STD_MIN_DB = 2.5        # speech/whisper fluctuate (syllables); steady noise does not
HARM_SPEECH = 0.45            # harmonicity at/above this: voiced speech
WHISPER_MAX_CENTROID_HZ = 4200  # rustling/hiss is brighter than whispers

# ---- audio event (temporal) -------------------------------------------------
EVENT_COVERAGE_S = 3.0   # speech/whisper must cover this much...
EVENT_WINDOW_S = 10.0    # ...within this sliding window
EVENT_COOLDOWN_S = 8.0

# ---- fusion -----------------------------------------------------------------
VIDEO_CUE_WEIGHTS = {
    "look_left": 0.6, "look_right": 0.6, "leaning": 0.5,
    "hand_to_face": 0.7, "lip_activity": 0.7,
}
AUDIO_HOLD_S = 4.0            # an audio event keeps counting as context this long
FUSION_ALERT_RISK = 0.6       # >= this: severity "alert", else "review"
FUSION_COOLDOWN_S = 8.0
AUDIO_ONLY_CAP = 0.4          # audio alone can never exceed this risk
FUSION_WEIGHTS = (0.5, 0.3, 0.2)   # audio conf, video cue, co-occurrence bonus

# ---- lips (optional video cue) ---------------------------------------------
LIP_ACTIVITY_SCALE = 0.05     # std of mouth opening that maps to activity 1.0
LIP_ACTIVITY_THRESHOLD = 0.35
LIP_WINDOW_S = 1.0