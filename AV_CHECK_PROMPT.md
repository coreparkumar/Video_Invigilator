# AI Prompt: Check and Add Voice + Video Test Code (with Audio Calibration)

Use this when you want an AI agent to (1) **audit** whether the voice-detection and audio-calibration test code is already in the project, and (2) **add or modify** whatever is missing, so you can test voice detection together with the video check on your own camera and microphone.

**How to use in OpenCode:** in the project folder run `/av-check` (the prompt is installed in `.opencode/commands/av-check.md`). Stay in **Plan** mode for Phase 1, review the audit table, then switch to **Build** mode and reply `GO`. In any other AI tool, paste the block below after telling it to read `AGENTS.md` (or paste the rules from it).

**What I found when I audited this repo myself** (so you can sanity-check the agent's answer):

| Area | Current state |
|---|---|
| `gesture_poc_av.py` with keys `c n m s q`, `--wav`, `--device`, `--lips` | Present |
| `AudioEngine` (recalibrate, mute, step) and 5 s noise-floor calibration | Present |
| `tools/audio_meter.py`, `audio_replay.py`, `audio_selftest.py` | Present |
| Privacy scan (no audio writes, no network, no speech-to-text) | Clean |
| Calibration **quality** check (dead mic / noisy / too loud / clipping) | **Missing** |
| Input level + clipping indicator on the overlay | **Missing** |
| Guided live test covering calibration + audio + video + fused + negative controls, with a report | **Missing** |

So the prompt below should come back with those three gaps and implement them with tests, but it must verify rather than trust this table.

---

## The Prompt

```text
TASK: Verify and, where needed, add code to test VOICE (whisper/talking) detection TOGETHER WITH VIDEO, including calibration of the audio input. Work in three phases. Read AGENTS.md first and obey it.

HARD RULES
- Do NOT modify gesture_poc.py. Existing tests must keep passing. No new dependencies, no network, no pip install.
- Privacy: no transcription, no speaker identification, never write audio to disk. Reports may contain numbers and labels only.
- You cannot use a real microphone or camera here. Write code + unit tests using fake sources, and clearly mark anything live as UNTESTED. Never invent live results.
- Every new threshold goes in audio/config.py and is marked UNTUNED.

PHASE 1: AUDIT (read-only, no edits). Inspect the repo and run `python -m pytest -q`. Report a table: ID | requirement | status (PRESENT / PARTIAL / MISSING) | evidence (file:line or command output).
 A1 gesture_poc_av.py exists, reuses gesture_poc.py without editing it, has keys c n m s q and flags --camera --out --no-audio --device --wav --lips
 A2 audio/engine.py: AudioEngine with step(), recalibrate(), toggle_mute(), calibrating, muted
 A3 audio calibration: CALIBRATION_S noise-floor calibration in AudioPipeline, with slow safe adaptation
 A4 calibration QUALITY check (detects: dead/muted mic, noisy room or talking during calibration, too-loud input, clipping) and a message shown to the user
 A5 live input level and clipping indicator (peak) exposed by the engine and drawn on the overlay
 A6 a guided live test that exercises audio calibration, video calibration, audio-only, video-only and FUSED detection plus negative controls, and writes a report
 A7 unit tests for A4 to A6 using synthetic audio (audio/synth.py) and scripted video cues
 A8 privacy scan: grep audio/, tools/, gesture_poc_av.py for audio writes (wave write mode, soundfile write), sockets/requests/urllib, speech-recognition libraries; report findings
 A9 tools/audio_meter.py and tools/audio_replay.py exist and handle a missing microphone with a friendly error
(Hint, verify do not trust: I expect A4, A5, A6, A7 to be MISSING and the rest PRESENT.)
End Phase 1 with: the list of gaps, the files you would create or change, and any risks. Then STOP and wait for my "GO". Do not edit anything before GO.

PHASE 2: IMPLEMENT only what Phase 1 marked MISSING or PARTIAL, in this order. After each item run its tests and print a 3-line status. If a test fails twice, print HALT and stop.

V1 Calibration quality (pure, in audio/baseline.py)
 - Add constants to audio/config.py: CAL_MAX_STD_DB=4.0, CAL_MIN_LEVEL_DB=-100.0, CAL_MAX_LEVEL_DB=-25.0, CLIP_PEAK=0.98 (UNTUNED; check them against synth levels, where ambient(-62 dBFS) gives a speech-band level near -60 dB, and justify or adjust).
 - `calibration_quality(frame_db, peak) -> (status, message)`; status priority: NO_SIGNAL (mean < CAL_MIN_LEVEL_DB: muted or wrong device), CLIPPING (peak >= CLIP_PEAK), TOO_LOUD (mean > CAL_MAX_LEVEL_DB), NOISY (std > CAL_MAX_STD_DB: someone spoke or noise changed during calibration), else OK. Messages tell the user what to do ("press n and stay quiet for 5 s").
 - AudioPipeline tracks the max |sample| during calibration, stores `calibration_status` and `calibration_message` when calibration finishes.
 - Tests: ambient -> OK; all zeros -> NO_SIGNAL; ambient with a whisper inside -> NOISY; loud steady noise -> TOO_LOUD; samples at +/-1.0 -> CLIPPING; pipeline sets the status after exactly 10 hops.

V2 Input level and clipping
 - AudioPipeline exposes `input_peak` (peak of the most recent push) and `clipping` (>= CLIP_PEAK). AudioEngine exposes input_peak, clipping, calibration_status, calibration_message.
 - gesture_poc_av.py overlay: show the calibration message in amber when status != OK, a red "CLIPPING" tag when clipping, and keep the existing keys. Minimal edit to that file only.
 - Tests with FakeAudioSource: clipped signal sets clipping; normal signal does not.

V3 Guided live check (testable core + thin live wrapper)
 - `audio/avcheck.py` (pure, no camera, no microphone, no clock reads: clock and sleep injected): `Stage(name, instruction, seconds, expect)` and `AVCheckRunner(engine, get_video_active, clock)` with `run(stages) -> list[StageResult(name, status PASS|FAIL|SKIPPED, measurements: dict, note)]`. Per step it calls engine.step(now, video_active) and records: audio label counts, audio events, alerts (severity, risk, latency from the first qualifying window), and each video cue with its longest continuous duration. `expect` supports: audio_event (bool), no_alert (bool), alert_severity ("alert"), video_cue (name, min_seconds), no_audio_event (bool), calibration_ok (bool).
 - Default stages: (1) devices ok: camera frames and microphone samples arrive; (2) audio calibration 5 s quiet, expect calibration_ok; (3) video calibration: pose visible; (4) NEGATIVE control: silent and still 15 s, expect no_alert and no_audio_event; (5) audio-only: whisper continuously 6 s, expect audio_event; (6) normal speech 6 s, expect audio_event; (7) video-only: look left and hold 3 s, expect video_cue look_left >= 1.5 s, then hand raised 2 s; (8) FUSED: whisper while looking left 8 s, expect alert_severity "alert"; (9) burst control: cough twice or rustle paper, expect no_audio_event.
 - `tools/av_check.py`: live wrapper. Opens the camera via gesture_poc.open_camera and the microphone via audio.capture.MicSource (friendly error and SKIPPED stages if one is missing; `--no-video`, `--no-audio`, `--device`, `--camera`, `--stages 2,5,8`). Shows the instruction and a countdown for each stage on the console and in the video window, waits for a keypress to begin each stage. Writes `docs/AV_CHECK_REPORT.md`: date, device names, a snapshot of the audio config constants, per-stage PASS/FAIL/SKIPPED with measurements and latency, calibration status/message, a list of what was UNTESTED, and a caution that one run is not accuracy evidence. Nothing but numbers and labels is saved.
 - Tests (no hardware): FakeAudioSource scenarios built with audio/synth.py plus scripted video-cue functions: negative control PASSes on ambient and FAILs when a whisper is injected; the whisper stage PASSes; the fused stage PASSes with a scripted look_left cue and FAILs without it; the burst control PASSes with a cough; a stage with a missing source is SKIPPED, not PASSED; the report file is written to tmp_path and the directory contains only the .md (no audio files).

V4 Scripts and docs
 - `scripts/av_check.bat`: runs the V1-V3 tests, then offers to launch `python tools/av_check.py`.
 - Add a section "Testing voice with video" to AUDIO_EXTENSION.md: how to run the live check, how to read the report, and what to do on NOISY / NO_SIGNAL / CLIPPING.

PHASE 3: HUMAN TEST SCRIPT. Print a numbered checklist I can follow with my real camera and microphone (the 9 stages above, what to do, what a PASS looks like, and what to try if it fails), then print the CHECKPOINT REPORT from AGENTS.md and STOP.
```

---

## What You Will Do Yourself After the AI Finishes (live test)

The AI cannot touch your camera or microphone, so these results only exist once you run them:

1. `scripts\av_check.bat` (or `python tools/av_check.py`). Sit where a candidate would sit, at normal exam distance.
2. **Calibration:** stay quiet for 5 s. Read the status. `NO_SIGNAL` = wrong/muted device (try `--device N`); `NOISY` = someone spoke, press `n` and retry; `TOO_LOUD` = lower the input gain; `CLIPPING` = gain too high.
3. **Negative control:** be still and quiet for 15 s. Any alert here is a false alarm: record it.
4. **Audio-only:** whisper for 6 s, then speak normally for 6 s. If the whisper is missed, move closer and note the distance at which it starts working.
5. **Video-only:** turn your head left and hold; raise a hand.
6. **Fused:** whisper while looking left. Expect an `alert` with both reasons (audio kind + visual cue).
7. **Burst control:** cough twice or rustle paper. Expect no audio event.
8. Open `docs/AV_CHECK_REPORT.md`, keep it as evidence, and repeat in a noisier condition (fan on, typing).

**Interpreting results:** a single clean run only shows the pipeline works on you in one room. Real accuracy needs several people, distances and noise conditions (see `AUDIO_EXTENSION.md` step 9 and the A9/A10 prompts), with consent (`docs/AUDIO_ETHICS.md`).

## Tips
- If the agent skips the audit and starts coding, reply: `STOP. Do Phase 1 only and wait for GO.`
- If it claims a live result: `You cannot run the mic or camera. Mark that UNTESTED.`
- To undo a bad edit: `/undo`.
