# OpenCode Prompt Pack: Add Audio (Whisper / Talking) Detection

Step-by-step, AI-driven implementation of the audio extension for Edge Invigilator, written for **[OpenCode](https://opencode.ai)** (open-source terminal AI coding agent). Every step is a ready slash command with a plan-first protocol, hard scope limits, an acceptance test, an explicit STOP condition and a clear CONTINUE trigger.

**What you get in the repo**

| File | Purpose |
|---|---|
| `AGENTS.md` | Project rules OpenCode loads automatically (scope, privacy, honesty, checkpoint report, auto-halt) |
| `opencode.json` | Permissions: edit = ask, tests allowed, `git push`, `curl`, `rm -rf` denied |
| `.opencode/commands/a0.md` ... `a10.md` | One slash command per step: `/a0` ... `/a10` |
| `.opencode/commands/audio-*.md` | Control commands: `/audio-halt`, `/audio-status`, `/audio-check`, `/audio-review`, `/audio-rescue` |
| `scripts\a0_*.bat` ... `a10_*.bat` | Same acceptance checks as one-click Windows scripts |

**Honest status:** a working reference implementation of A0 to A8 exists in this repo (42 tests and a synthetic self-test pass). A9 and A10 are new prompt-only steps. I could not run OpenCode itself here, so the config follows OpenCode's documented conventions (`AGENTS.md`, `opencode.json`, `.opencode/commands/`), but verify two details against your installed version: the commands folder name (`commands/`, older builds used `command/`) and the `permission` syntax (https://opencode.ai/docs/permissions).

---

## 1. Choose a Mode

**Mode A: Practice rebuild (recommended if you want to learn / produce your own repo history).** Move the reference aside so OpenCode builds from scratch, and use it only to compare:
```
git init  &&  git add -A  &&  git commit -m "baseline with reference audio"
mkdir reference
git mv audio reference\audio
git mv tools reference\tools
git mv tests reference\tests
git mv gesture_poc_av.py reference\gesture_poc_av.py
git commit -m "move reference aside"
```
Then run `/a0` to `/a8`. After each step compare your result with `reference\` (for example `fc audio\detector.py reference\audio\detector.py`). Your tests, not an exact match, are the gate.

**Mode B: Extend.** Keep the code as is, run `scripts\status_audio.bat`, and only use `/a9` (validation tools) and `/a10` (ablation harness), plus `/audio-review` on the existing code.

## 2. One-Time Setup

1. Install OpenCode and connect a model: `curl -fsSL https://opencode.ai/install | bash` (or `npm i -g opencode-ai`), run `opencode`, then `/connect` and `/models`.
2. `cd` into the project, make sure it is a git repo (OpenCode's `/undo` relies on git).
3. Keep the supplied `AGENTS.md` (do not let `/init` overwrite it; if you run `/init`, merge rather than replace).
4. Install dependencies: `build.bat` (adds `sounddevice`), or `pip install -r requirements.txt`. Confirm `sounddevice` with me first only if you are using the strict prompts (A1 asks before adding it).
5. Open OpenCode in the project folder: `opencode`.

## 3. The Loop (repeat for every step)

```
Tab to PLAN agent  ->  /aN  ->  read the plan  ->  (adjust or reply "GO")
Tab to BUILD agent ->  say "GO"  ->  agent edits (approve edits) and runs tests
/audio-check aN    ->  /audio-review  ->  run scripts\aN_*.bat yourself
PASS?  yes -> git commit -> /new (fresh session) -> next /aN+1
       no  -> /audio-rescue  (or /undo)
```
Why: Plan mode cannot edit, so you always see the plan first; a fresh session per step keeps context small and the agent on scope; a commit per step gives you restore points.

## 4. Control Commands (STOP and CONTINUE)

| You do | Effect |
|---|---|
| `/aN` | **CONTINUE trigger**: start step N (only after the previous step passed and you committed) |
| `GO` | Approve the plan and let the Build agent edit |
| `/audio-halt` (or type `STOP`) | **STOP**: no more edits, print current state |
| `/audio-status` | Which steps appear complete, what is next |
| `/audio-check aN` | Run the step's tests and report PASS/FAIL, no edits |
| `/audio-review` | Strict review of uncommitted changes against `AGENTS.md`; verdict SAFE TO CONTINUE or FIX FIRST |
| `/audio-rescue` | Diagnose without editing; propose the cheapest experiment and minimal fix |
| `/undo` / `/redo` | OpenCode built-ins: revert / restore the last change set |
| `/new`, `/compact` | Fresh session / shrink context |

**Auto-halt:** the agent must print `HALT: <reason>` and stop on repeated test failure, a new dependency, network call or out-of-scope file, a diff well over plan, ambiguity, or anything that could save audio or transcribe speech.

**Typing tips:** reference files with `@audio/config.py`; the commands already tell the agent to read `AGENTS.md`.

## 5. Step Overview

| Step | Command | Builds | Acceptance | Windows script |
|---|---|---|---|---|
| A0 | `/a0` | Ethics and consent gate | `(no code) read the file` | `scripts\a0_ethics.bat` |
| A1 | `/a1` | Config and audio capture | `python -m pytest tests/test_audio_capture.py -q` | `scripts\a1_capture.bat` |
| A2 | `/a2` | Audio features and synthetic signals | `python -m pytest tests/test_audio_features.py -q` | `scripts\a2_features.bat` |
| A3 | `/a3` | Noise-floor calibration | `python -m pytest tests/test_audio_baseline.py -q` | `scripts\a3_baseline.bat` |
| A4 | `/a4` | Speech / whisper detector (rules) | `python -m pytest tests/test_audio_detector.py -q` | `scripts\a4_detector.bat` |
| A5 | `/a5` | Audio event session (temporal logic) | `python -m pytest tests/test_audio_session.py -q` | `scripts\a5_session.bat` |
| A6 | `/a6` | Streaming pipeline and synthetic self-test | `python -m pytest tests/test_audio_pipeline.py -q ` | `scripts\a6_pipeline.bat` |
| A7 | `/a7` | Fusion and lip cue | `python -m pytest tests/test_audio_fusion.py tests/test_audio_lips.py -q` | `scripts\a7_fusion.bat` |
| A8 | `/a8` | Engine and live integration | `python -m pytest tests -q -k audio ` | `scripts\a8_integration.bat` |
| A9 | `/a9` | Real-world validation tools | `python -m pytest tests/test_audio_eval.py -q` | `scripts\a9_validation.bat` |
| A10 | `/a10` | Ablation harness (video-only vs audio-only vs fused) | `python -m pytest tests/test_ablation.py -q` | `scripts\a10_ablation.bat` |

Dependency order is fixed: config, then features, baseline, detector, session, pipeline, fusion, engine, app. Never skip ahead.

---

## 6. The Prompts

Each block is exactly what the slash command sends. If you prefer not to use slash commands, paste the block into OpenCode.

### A0. Ethics and consent gate

**Run:** `/a0` in Plan mode, review, then `GO` in Build mode. **Scope:** docs/AUDIO_ETHICS.md

```text
STEP A0: Ethics and consent gate.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): docs/AUDIO_ETHICS.md

Create `docs/AUDIO_ETHICS.md`. No code in this step.
Sections: (1) design commitments: detects acoustic activity only, NO transcription, NO speaker identification or voiceprints, no audio written to disk, nothing sent over a network, a flag is never proof, audio alone is low severity; (2) a tick-box checklist to complete BEFORE recording any person: ethics approval reference, legal check of audio-recording rules in my jurisdiction (blank for me to fill), signed informed consent, signage, retention period, access-controlled storage with no cloud upload, a mute procedure for announcements and invigilator speech, an appeals/human-review procedure, a fairness check across voices, accents, languages and distances; (3) known limitations: whispers may be inaudible to a laptop mic beyond short range, ventilation/cough/paper noise cause errors, one microphone cannot tell who spoke, thresholds untuned until evaluated in my room.
Add one line stating that synthetic audio and my own voice alone need no approval.

ACCEPTANCE: (no code) read the file
STOP AND ASK ME IF: the document suggests recording people without consent, transcription, or voiceprints.
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `(no code) read the file`, then `/audio-check a0` and `/audio-review`
- **STOP if:** the document suggests recording people without consent, transcription, or voiceprints.
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a1`

### A1. Config and audio capture

**Run:** `/a1` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/__init__.py, audio/config.py, audio/capture.py, tests/conftest.py, tests/test_audio_capture.py

```text
STEP A1: Config and audio capture.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/__init__.py, audio/config.py, audio/capture.py, tests/conftest.py, tests/test_audio_capture.py

Create the files listed in scope.

1. `audio/__init__.py`: docstring stating the privacy rules (no transcription, no speaker ID, no audio saved, no network).
2. `audio/config.py` must contain EXACTLY these constants (copy verbatim; they are the single source of truth for every later step):
```python
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
```
3. `audio/capture.py`:
   - `list_input_devices()`: lazy-import `sounddevice`; return `[(index, name, max_input_channels)]` for devices that can record.
   - `MicSource(device=None, sr=SAMPLE_RATE, max_chunks=200)`: lazy-import sounddevice; mono float32 InputStream whose callback copies each block into `queue.Queue(maxsize=max_chunks)`; when full, drop the OLDEST block, count it in `self.dropped`, never block. `read()` returns all queued samples concatenated (empty float32 array if none). `finished` is always False. `close()` is idempotent.
   - `FakeAudioSource(samples, chunk=FRAME)`: `read()` returns the next chunk (empty after the end), property `finished`, `close()`.
   - `load_wav(path, sr=SAMPLE_RATE)`: stdlib `wave` only; 16-bit PCM else ValueError; stereo to mono by mean; linear resample with `np.interp`; float32 in [-1, 1].
   - `WavSource(path, sr, clock=time.monotonic)`: `read()` returns the samples that have elapsed in real time since the first call (injectable clock for tests); `finished`.
4. `tests/conftest.py`: put the repo root on `sys.path`.
5. Tests, no microphone needed: FakeAudioSource on 1500 samples gives chunk sizes [512, 512, 476] then empty; `load_wav` on a temporary 8 kHz stereo file returns 16 kHz mono, length within +/-1 sample, peak preserved; `WavSource` with a fake clock returns 0, then 8000 samples after 0.5 s, then the rest and `finished`; a non-16-bit file raises ValueError.
Do NOT pip install. Ask me before adding `sounddevice==0.4.7` to requirements.txt.

ACCEPTANCE: python -m pytest tests/test_audio_capture.py -q
STOP AND ASK ME IF: any code writes audio to disk or the network, or you need to pip install anything.
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_capture.py -q`, then `/audio-check a1` and `/audio-review`
- **STOP if:** any code writes audio to disk or the network, or you need to pip install anything.
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a2`

### A2. Audio features and synthetic signals

**Run:** `/a2` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/features.py, audio/synth.py, tests/test_audio_features.py

```text
STEP A2: Audio features and synthetic signals.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/features.py, audio/synth.py, tests/test_audio_features.py

Pure NumPy, no I/O, no clocks. Use constants from `audio/config.py` only.

`audio/features.py`
- `EPS = 1e-12`, `_WIN = np.hanning(FRAME)`.
- `frame_harmonicity(frame, sr, fmin, fmax)`: subtract the mean; if energy < 1e-10 return 0.0; autocorrelation via FFT (zero-pad to 2n), normalize by lag 0; consider lags `int(sr/fmax)` to `min(int(sr/fmin), n-1)`; divide each lag's value by `(n-lag)/n` to undo the triangular bias; return the max clipped to [0, 1].
- `window_features(window, sr)`: split into whole FRAME-sample frames (ValueError if fewer than one frame); `spec = rfft(frame * _WIN) / (_WIN.sum()/2)`; `power = |spec|^2`. Return a dict with: `frame_speech_db` (per frame, 10*log10(sum of power in SPEECH_BAND + EPS)), `frame_harm` (per-frame harmonicity), `flatness` (geometric/arithmetic mean of the window-averaged power over FLATNESS_BAND), `centroid_hz` (power-weighted mean frequency over the same band), `rms_db` (20*log10(rms+EPS)), `crest` (peak/rms), `zcr`.

`audio/synth.py` (test signals; all seeded, all return float arrays at 16 kHz):
- `_scale(x, db)` scales to the given RMS in dBFS.
- `ambient(sec, db=-62, seed=0)`: white noise.
- `_bandpass_noise(n, lo, hi, seed)`: FFT-mask noise, then divide the spectrum by `sqrt(max(f,100)/100)` (gentle tilt).
- `_syllables(n, rate_hz=4.0)`: envelope `0.15 + 0.85*0.5*(1+sin(2*pi*rate*t))`.
- `speech(sec, db=-35, f0=130)`: sum of harmonics k=1..int(3800/f0)-1 with amplitude 1/k and a 3 Hz, +/-2% vibrato on f0, times `_syllables`.
- `whisper(sec, db=-38)`: bandpass noise 500-4500 Hz times `_syllables`.
- `cough(sec=0.25, db=-30)`: white noise times `exp(-t/(0.06*16000))`.
- `steady_noise(sec, db=-40)`: white noise.
- `paper_rustle(sec, db=-38)`: bandpass 3500-8000 Hz times `_syllables(rate_hz=9)`.
- `over(background, event, at_s=0.0)`: add event onto a copy of background.

Tests (`tests/test_audio_features.py`): a 150 Hz sine has harmonicity > 0.8 and white noise < 0.35; an all-zero window gives finite values everywhere and zero harmonicity; `synth.whisper` has centroid between 800 and 3500 Hz and mean harmonicity < 0.4; raising the level by 10 dB raises mean `frame_speech_db` by 9 to 11 dB; a window shorter than one frame raises ValueError.

ACCEPTANCE: python -m pytest tests/test_audio_features.py -q
STOP AND ASK ME IF: NaN or infinity appears in any feature, or a feature needs a new dependency.
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_features.py -q`, then `/audio-check a2` and `/audio-review`
- **STOP if:** NaN or infinity appears in any feature, or a feature needs a new dependency.
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a3`

### A3. Noise-floor calibration

**Run:** `/a3` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/baseline.py, tests/conftest.py (add fixtures), tests/test_audio_baseline.py

```text
STEP A3: Noise-floor calibration.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/baseline.py, tests/conftest.py (add fixtures), tests/test_audio_baseline.py

`audio/baseline.py`: class `NoiseFloor`.
- Attributes `mean_db=None`, `std_db=0.0`; property `calibrated`.
- `calibrate(frame_db)`: mean and std of the array (ValueError if empty).
- Properties `margin_db = max(MIN_MARGIN_DB, MARGIN_STD_MULT * std_db)` and `threshold_db = mean_db + margin_db`.
- `adapt(level_db, dt, label)`: do nothing unless calibrated AND `label == "silence"` AND `abs(level_db - mean_db) <= FLOOR_GATE * margin_db`; otherwise move `mean_db` toward `level_db` with `alpha = 1 - exp(-dt / FLOOR_TAU_S)`. Sustained whispering must never be learned as ambient.

Add to `tests/conftest.py`: a `floor` fixture that calibrates a NoiseFloor on 6 s of `synth.ambient(6, seed=10)` (1 s windows, 0.5 s hop, concatenating `frame_speech_db`), and a helper `feats_for(signal)` returning `window_features` of the last 1 s.

Tests: calibrated floor is around -60 dB with margin >= 8; +3 dB ambient drift is absorbed after 5 simulated minutes (600 updates of 0.5 s); a +20 dB level fed as "whisper" or as "silence" never moves the floor (change < 0.01 dB); recalibration replaces everything; `adapt` on an uncalibrated floor is a no-op.

ACCEPTANCE: python -m pytest tests/test_audio_baseline.py -q
STOP AND ASK ME IF: the sustained-whisper-is-not-absorbed test fails (that is a cheating hole; ask me before changing the gate).
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_baseline.py -q`, then `/audio-check a3` and `/audio-review`
- **STOP if:** the sustained-whisper-is-not-absorbed test fails (that is a cheating hole; ask me before changing the gate).
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a4`

### A4. Speech / whisper detector (rules)

**Run:** `/a4` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/detector.py, tests/test_audio_detector.py

```text
STEP A4: Speech / whisper detector (rules).
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/detector.py, tests/test_audio_detector.py

`audio/detector.py`: `classify_window(feats, floor) -> (label, confidence, info)`. Pure. Thresholds come ONLY from `audio/config.py`.
Algorithm, in this exact order:
1. `active = frame_speech_db > floor.threshold_db`; `frac = active.mean()`; `info` has `level_db`, `active_frac`, `above_floor_db`.
2. If `frac == 0`: return `("silence", 1.0, info)`.
3. `over = mean(active frame dB) - floor.mean_db`; `conf = 0.4 + 0.6 * clip((over - floor.margin_db) / 12, 0, 1)`; `level_std = std(active frame dB)`; `harm = median(frame_harm[active])`; add `level_std_db`, `harmonicity`, `centroid_hz` to info.
4. `frac < ACTIVE_FRAC_SUSTAINED` -> "noise_burst" (cough, door, chair).
5. `level_std < LEVEL_STD_MIN_DB` -> "noise" (steady hum, no syllable rhythm).
6. `harm >= HARM_SPEECH` -> "speech".
7. `centroid_hz > WHISPER_MAX_CENTROID_HZ` -> "noise" (bright hiss, paper).
8. otherwise -> "whisper".

Tests (table-driven, use the `floor` fixture and `synth`, each scenario is ambient(1.2, seed=21) plus the event): ambient only -> silence; speech -> speech; whisper -> whisper; cough placed at 0.5 s -> noise_burst; steady_noise -> noise; paper_rustle -> noise; near-miss `whisper(db=-68)` -> silence; a louder whisper (db=-35) has higher confidence than a quieter one (db=-50), both labelled whisper; confidence always in [0, 1].
State clearly in the report that accuracy is only proven on synthetic signals.

ACCEPTANCE: python -m pytest tests/test_audio_detector.py -q
STOP AND ASK ME IF: a class needs a threshold outside config.py, or a near-miss test is flaky (the threshold semantics need my decision).
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_detector.py -q`, then `/audio-check a4` and `/audio-review`
- **STOP if:** a class needs a threshold outside config.py, or a near-miss test is flaky (the threshold semantics need my decision).
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a5`

### A5. Audio event session (temporal logic)

**Run:** `/a5` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/session.py, tests/test_audio_session.py

```text
STEP A5: Audio event session (temporal logic).
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/session.py, tests/test_audio_session.py

`audio/session.py`: pure; time is always passed in as `now`.
- `@dataclass AudioEvent(kind, duration, mean_conf, start, end)`; `kind` is "whispering" or "talking".
- `AudioSession(hop_s=HOP_S)` with `muted=False`, a deque of `(t, label, conf)`, `_last_event = -inf` (NOT 0, so cooldown never blocks the first event).
- `update(label, conf, now) -> list[AudioEvent]` (0 or 1 item): if muted, clear history and return []. Otherwise append, drop entries older than EVENT_WINDOW_S, count entries labelled "speech" or "whisper"; `coverage = count * hop_s`. If `coverage >= EVENT_COVERAGE_S` and `now - _last_event >= EVENT_COOLDOWN_S`: emit an event with `kind="whispering"` when at least half of the qualifying entries are "whisper", else "talking"; `duration=coverage`; `mean_conf` = average confidence of qualifying entries; `start` = time of the first qualifying entry; `end = now`.
- "noise_burst", "noise" and "silence" never count.

Tests (FakeClock-style timestamps starting at 1000.0, hop 0.5): a cough or brief whisper does not fire; 8 whisper windows fire exactly once with kind whispering and duration >= 3; 7 speech windows give kind talking; a second run of 8 whisper windows 4 s later is blocked by cooldown, and a third later run fires; alternating whisper/silence for 6 s fires (3 s coverage) but whisper + 3x silence repeated 4 times does not; muted never fires.

ACCEPTANCE: python -m pytest tests/test_audio_session.py -q
STOP AND ASK ME IF: any test needs sleep() or reads the system clock.
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_session.py -q`, then `/audio-check a5` and `/audio-review`
- **STOP if:** any test needs sleep() or reads the system clock.
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a6`

### A6. Streaming pipeline and synthetic self-test

**Run:** `/a6` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/pipeline.py, tools/audio_selftest.py, tests/test_audio_pipeline.py

```text
STEP A6: Streaming pipeline and synthetic self-test.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/pipeline.py, tools/audio_selftest.py, tests/test_audio_pipeline.py

`audio/pipeline.py`
- `@dataclass WindowResult(t, label, conf, level_db, info)`.
- `AudioPipeline(t0=0.0, sr=SAMPLE_RATE, auto_calibrate=True)`: window = `WINDOW_S*sr` samples, hop = `HOP_S*sr`. Keep a float32 buffer with an absolute start index; the next window ends at absolute sample `window`, then every `hop`. `push(samples) -> list[WindowResult]` returns one result per completed hop and trims the buffer. Timestamp of a window = `t0 + end_index / sr` (deterministic, no system clock).
- Calibration: `start_calibration()` sets `calibrating=True` and collects each window's `frame_speech_db`; after `CALIBRATION_S` worth of hops (10 hops at 0.5 s) call `floor.calibrate(concatenated)` and set `calibrating=False`. Results during calibration have label "calibrating" (including the 10th).
- After calibration: `classify_window`, then `floor.adapt(level, HOP_S, label)` where level is the mean of `frame_speech_db`.

`tools/audio_selftest.py`: build 6 s of ambient (calibration) then segments, each over `ambient(seed=20+i)`: ambient 4 s -> silence; whisper 6 s -> whisper; ambient 3 -> silence; speech 6 -> speech; ambient 3 -> silence; steady hum 6 -> noise; ambient 3 -> silence; paper rustle 6 -> noise. Feed the pipeline one hop at a time, timing `push` with `perf_counter`. For each segment ignore the first 1.2 s (window straddles the boundary), take the majority label, PASS if it matches and its share >= 80%. Print a table, the mean/max compute ms per hop against the 500 ms budget, and `RESULT: PASS|FAIL`; exit code 0 or 1.

Tests: the first 10 results are "calibrating" and later ones are classified; "whisper" appears for a whisper scenario; consecutive timestamps are 0.5 s apart starting at `t0 + 1.0`; feeding 700-sample chunks gives exactly the same label sequence as feeding everything at once; running the pipeline creates no files (use tmp_path + chdir).

ACCEPTANCE: python -m pytest tests/test_audio_pipeline.py -q  AND  python tools/audio_selftest.py (exit code 0)
STOP AND ASK ME IF: compute per 0.5 s hop exceeds 100 ms on my machine (tell me before optimizing).
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_pipeline.py -q  AND  python tools/audio_selftest.py (exit code 0)`, then `/audio-check a6` and `/audio-review`
- **STOP if:** compute per 0.5 s hop exceeds 100 ms on my machine (tell me before optimizing).
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a7`

### A7. Fusion and lip cue

**Run:** `/a7` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/fusion.py, audio/lips.py, tests/test_audio_fusion.py, tests/test_audio_lips.py

```text
STEP A7: Fusion and lip cue.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/fusion.py, audio/lips.py, tests/test_audio_fusion.py, tests/test_audio_lips.py

`audio/fusion.py`
- `@dataclass Alert(t, risk, severity, reasons: list, audio_kind: str, video_cues: list)`; severity is "alert" or "review".
- `Fusion` keeps `_audio_until=-inf`, `_audio_conf`, `_audio_kind`, `_last_fused=-inf`.
- `update(audio_events, video_active, now, lip_activity=0.0) -> list[Alert]`: `cues` = sorted keys of `video_active` that exist in VIDEO_CUE_WEIGHTS, plus "lip_activity" if `lip_activity >= LIP_ACTIVITY_THRESHOLD`.
  - For each new audio event: set `_audio_until = now + AUDIO_HOLD_S`, store conf/kind; if there are no cues emit a "review" Alert with `risk = min(AUDIO_ONLY_CAP, 0.4 * conf)` and reason "<kind> for X.Xs (audio only, no visual cue)".
  - If `now < _audio_until` and cues exist and `now - _last_fused >= FUSION_COOLDOWN_S`: `risk = min(1, wa*conf + wv*max(cue weight) + wb)` with `FUSION_WEIGHTS`; severity "alert" if `risk >= FUSION_ALERT_RISK` else "review"; reasons list the audio kind with confidence and the visual cues; this fused alert replaces any audio-only review from the same call.
- Video-only never produces an alert here.

`audio/lips.py` (optional cue): `lip_opening(lm)` = |y(13) - y(14)| / distance(corner 61, corner 291), or None when landmarks are missing or width < 1e-6. `LipTracker.update(opening, now)` keeps LIP_WINDOW_S of openings and returns `clip(std / LIP_ACTIVITY_SCALE, 0, 1)` (0 with fewer than 5 samples). `FaceMeshLips` lazily imports `mediapipe` (`mp.solutions.face_mesh.FaceMesh(max_num_faces=1)`), `opening(rgb)` and `close()`.

Tests: audio only -> one "review" with risk <= 0.4; video cue only -> []; audio + look_right -> one "alert" with risk >= 0.6, reasons mention look_right; a cue 2 s after an audio event still fuses; a cue 10 s after does not; "hand_raised" is ignored; lip_activity 0.9 counts as a cue and the fused cooldown blocks a second alert 1 s later. Lips: opening is scale invariant; missing landmarks -> None; a flapping mouth gives activity > 0.5 and a closed mouth < 0.1.

ACCEPTANCE: python -m pytest tests/test_audio_fusion.py tests/test_audio_lips.py -q
STOP AND ASK ME IF: fusion lets audio alone exceed AUDIO_ONLY_CAP, or any code reads a clock.
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_fusion.py tests/test_audio_lips.py -q`, then `/audio-check a7` and `/audio-review`
- **STOP if:** fusion lets audio alone exceed AUDIO_ONLY_CAP, or any code reads a clock.
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a8`

### A8. Engine and live integration

**Run:** `/a8` in Plan mode, review, then `GO` in Build mode. **Scope:** audio/engine.py, gesture_poc_av.py (new, do not edit gesture_poc.py), tests/test_audio_pipeline.py (add engine tests)

```text
STEP A8: Engine and live integration.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): audio/engine.py, gesture_poc_av.py (new, do not edit gesture_poc.py), tests/test_audio_pipeline.py (add engine tests)

1. `audio/engine.py`: `AudioEngine(source, t0, enabled=True)` owning an `AudioPipeline`, `AudioSession` and `Fusion`. Properties `calibrating`, `muted`; methods `toggle_mute()`, `recalibrate()`, `close()` (tolerate `source=None`). `step(now, video_active=(), lip_activity=0.0) -> list[Alert]`: if disabled return []; push `source.read()` through the pipeline; for each result record `label/conf/level_db` and feed `session.update(label, conf, r.t)`; pass the collected events to `fusion.update(events, set(video_active), now, lip_activity)`; remember the last alert's risk and reasons. Initial label is "calibrating" (or "off" if disabled).
2. Add engine tests (FakeAudioSource): a 20 s scenario with a whisper from 8-15 s and a `look_left` cue active 13-16 s must produce one "alert" fused with kind "whispering" and cue "look_left"; disabled engine returns [] and label "off"; a muted engine never alerts.
3. `gesture_poc_av.py`: a NEW entry point that imports and reuses `open_camera`, `InvigilatorSession` and `render_overlay` from `gesture_poc.py` WITHOUT modifying that file. CLI: `--camera`, `--out evidence`, `--no-audio`, `--device N`, `--wav file`, `--lips`. If the microphone cannot be opened, print a warning and continue video-only. Per frame: mirror-flip, pose, `session.update`, then `engine.step(now, session.active, lip_activity)`. For each alert: save ONE snapshot JPEG and append a row to `evidence/av_events.csv` with columns `timestamp, severity, risk, audio_kind, video_cues, reasons, snapshot`; show a banner for 5 s. Draw an audio panel at top right (label colour, level bar above the noise floor, "calibrating: stay quiet", last risk). Keys: c posture calibrate, n audio recalibrate, m mute, s skeleton, q quit. Release everything in `finally`. Never store audio.
You cannot run the camera: say so in the report and give me the exact manual check.

ACCEPTANCE: python -m pytest tests -q -k audio  AND  python -m py_compile gesture_poc_av.py
STOP AND ASK ME IF: you need to change gesture_poc.py, or the app would save audio, or you cannot reuse its functions.
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests -q -k audio  AND  python -m py_compile gesture_poc_av.py`, then `/audio-check a8` and `/audio-review`
- **STOP if:** you need to change gesture_poc.py, or the app would save audio, or you cannot reuse its functions.
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a9`

### A9. Real-world validation tools

**Run:** `/a9` in Plan mode, review, then `GO` in Build mode. **Scope:** tools/audio_replay.py, tools/audio_meter.py, tools/audio_eval_sweep.py, docs/AUDIO_EVAL.md, tests/test_audio_eval.py

```text
STEP A9: Real-world validation tools.
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): tools/audio_replay.py, tools/audio_meter.py, tools/audio_eval_sweep.py, docs/AUDIO_EVAL.md, tests/test_audio_eval.py

1. `tools/audio_replay.py recording.wav`: load with `load_wav`, run `AudioPipeline` + `AudioSession` offline, print a timeline (time, label, confidence, dB above floor, EVENT markers) and a summary; warn if the file is shorter than the calibration period.
2. `tools/audio_meter.py [--device N] [--list]`: live microphone check; calibrate 5 s ("stay quiet"), then print the label and a level bar for every 0.5 s until Ctrl+C. Catch microphone/PortAudio errors and exit with a friendly message pointing to AUDIO_EXTENSION.md. Nothing is recorded.
3. `tools/audio_eval_sweep.py manifest.csv`: manifest columns `wav, expected, distance_m, noise, consent_ok, notes`. Refuse rows where `consent_ok` is not "yes". Each WAV must start with 5 s of quiet. For each file run the pipeline and compute the majority label over the active region and whether an audio event fired. Write `docs/AUDIO_EVAL.md`: a table of detection rate by `distance_m` x `noise`, per-class confusion for `expected` vs predicted, and false alarms per hour on files whose expected label is silence/typing. No claims beyond the data.
4. `docs/AUDIO_EVAL.md` template header explaining the recording protocol: whisper at 0.5, 1, 2, 3 m; normal speech; cough; paper rustle; typing; fan on/off; 5-minute quiet soak.
Tests build tiny WAVs with `synth` in tmp_path and check: manifest rows without consent are rejected, the table is produced, a silent file yields 0 events.

ACCEPTANCE: python -m pytest tests/test_audio_eval.py -q
STOP AND ASK ME IF: the tool would record or save raw audio, or a recording has no consent note in the manifest.
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_audio_eval.py -q`, then `/audio-check a9` and `/audio-review`
- **STOP if:** the tool would record or save raw audio, or a recording has no consent note in the manifest.
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> `/a10`

### A10. Ablation harness (video-only vs audio-only vs fused)

**Run:** `/a10` in Plan mode, review, then `GO` in Build mode. **Scope:** tools/eval_ablation.py, docs/ABLATION.md, tests/test_ablation.py

```text
STEP A10: Ablation harness (video-only vs audio-only vs fused).
Read AGENTS.md first. PLAN FIRST: in at most 8 bullets list the files you will create or edit, the tests you will write, and any risk or ambiguity. Then wait for my "GO" before changing any file.

SCOPE (edit only these): tools/eval_ablation.py, docs/ABLATION.md, tests/test_ablation.py

Your requirements document asks for a video-only vs audio-only vs fused comparison.
`tools/eval_ablation.py sessions.csv`: input has one row per second: `session, t, truth, audio_event, video_cue` (0/1 each; `truth` = a human-labelled suspicious-behaviour second). Three detectors per second: video-only = `video_cue`; audio-only = `audio_event`; fused = audio_event within the last AUDIO_HOLD_S seconds AND video_cue now. Compute per detector and overall: true/false positives, precision, recall, F1, and false alarms per hour (using the number of seconds). Group results by `session` and warn when there are fewer than 3 sessions or fewer than 20 truth seconds. Print a markdown table to stdout and write it to `docs/ABLATION.md` with a header stating the data source and a caution that thresholds must not be tuned on this same data.
Tests: a hand-made CSV where fused removes a known false positive gives higher precision than either single modality; division by zero is handled (no positives); the per-session grouping works.

ACCEPTANCE: python -m pytest tests/test_ablation.py -q
STOP AND ASK ME IF: the metrics are computed on data that was also used to tune thresholds (ask me to split it).
DONE WHEN: the acceptance command passes. Then print the CHECKPOINT REPORT from AGENTS.md and STOP. Do not start the next step.
```

- **Verify:** `python -m pytest tests/test_ablation.py -q`, then `/audio-check a10` and `/audio-review`
- **STOP if:** the metrics are computed on data that was also used to tune thresholds (ask me to split it).
- **CONTINUE when:** all tests pass, the review says SAFE TO CONTINUE, and you committed -> the project is complete

---

## 7. Rescue Prompts (paste when needed)

**Scope creep:** `STOP. You edited files outside this step's scope. List every file you changed that is not in scope, revert only those, and wait.`

**Tests failing twice:** `HALT. Do not edit. Paste the failing output, explain the root cause in 3 sentences, give two fixes with trade-offs, and wait for my choice.`

**Too big:** `Split this step into three sub-steps under 60 lines each with their own tests. Start only with sub-step 1 and wait.`

**Drift reset:** `Re-read AGENTS.md. Confirm in one line each that you will follow rules 1, 3, 4, 7 and 8. Then wait.`

**Explain before coding:** `Before writing code, explain your plan in 5 bullets, list the files and the tests, and wait for GO.`

**Tune a threshold safely (after A9 data):** `Change only <CONSTANT> in audio/config.py from X to Y. Re-run the whole test suite and tools/audio_selftest.py. Report any test that changes and do not edit tests to make them pass.`

## 8. Real-World Validation Checklist (after A8 and A9)

- [ ] `docs/AUDIO_ETHICS.md` checklist completed before recording any person.
- [ ] `tools/audio_meter.py --list`, pick the right microphone, and verify the meter reacts.
- [ ] Record whisper at 0.5, 1, 2, 3 m, normal speech, cough, paper, typing, fan on/off (with consent), then run `tools/audio_eval_sweep.py`.
- [ ] 5-minute quiet soak: zero alerts.
- [ ] Tune one constant at a time with the safe-tuning prompt above; keep tuning data separate from ablation data (A10).
- [ ] Report limits honestly (range, noise, false alarms) in your README.

## 9. Tips for Good Results With OpenCode

- Keep `AGENTS.md` short; it is sent with every request.
- One step per session; commit after each green step.
- Do not let the agent edit tests to make them pass: say so if it tries (rule 3 of the rescue prompts).
- Use a stronger model for A4 (detector) and A7 (fusion), which carry the judgment; a cheaper one is fine for A1, A2 and A9.
- If the agent invents a number you did not run, reply: `Mark that UNTESTED or run it.`
