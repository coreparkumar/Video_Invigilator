# Edge Invigilator: Adding Sound (Whisper/Talking) Detection

**Plan v0.1 | 6 October 2026** **Inputs:** your *Project Requirements for Decentralized Multimodal Edge AI* document, the Edge Invigilator POC (`gesture_poc.py`), the PRD, and the vibe-coding module plan.

---

## 1. What the Requirements Document Asks For (audio-relevant)

| Requirement (from your document) | What it means for the audio feature |
| --- | --- |
| Whispering or talking between candidates is a target violation, "using audio together with visual behavioural cues" | Audio is never the sole evidence. Audio events are fused with video context |
| Video + audio are the **mandatory two modalities** for the POC; RF is optional | Audio is a first-class modality, with its own pipeline, tests and ablation |
| AI is decision support only; final decision is human | Output is an alert with a risk score and evidence, never a verdict |
| A single behaviour (one head turn, brief speech) must not be classed as cheating | Temporal and contextual logic is required on audio too |
| Fully local, no cloud; raw audio/video not continuously transmitted | All audio processing on-device; send only compact events and scores |
| Latency target: event-to-alert at most 2 s after sufficient evidence | Define the audio observation window separately from compute latency; measure both |
| Success is measured by precision/recall/F1, false-positive rate, and **video-only vs audio-only vs fused ablation** | Build the evaluation harness for ablations from day one |
| Dataset: none exists; mock exam with consent and ethical approval | Plan a recording tool and a consent gate before any human audio is captured |
| Out of scope: face recognition, biometric profiling, automatic discipline | No speaker identification or voiceprints. No speech-to-text of what is said |

## 2. Design Decisions (and why)

1. **Detect acoustic activity, not content.** The system decides "someone is speaking or whispering" from signal features. It does **not** transcribe words and does **not** identify speakers. This respects the document's out-of-scope list and greatly reduces privacy risk.
2. **Metadata-only by default.** Raw audio stays in a short in-memory ring buffer. Only scores, timestamps and (optionally, with consent) a short evidence clip are saved when an alert fires.
3. **Rules first, learning second.** Start with a transparent, calibratable detector (noise floor + speech-band energy + whisper-likeness), then replace or augment it with a small learned classifier once mock-exam data exists. This mirrors the research approach in your document: baselines first, thresholds after experiments.
4. **Late fusion first.** Combine an audio risk score with video context (head turned toward a neighbour, lip movement, hand at mouth, leaning) using a weighted, auditable rule. Learned fusion only after data exists.
5. **A single microphone cannot say *who* spoke.** In the single-laptop POC, audio gives "talking activity in the room". Attribution to a candidate comes from video cues. Directionality (microphone array, multi-node arrival-time differences) is a later extension.

## 3. Honest Technical Limits (read before promising anything)

- **Whispers are hard.** They are low-energy, noise-like, and lack the pitch that normal speech has. At 3 m or more in a room with ventilation noise, a laptop microphone may not hear them reliably. Expect to measure detection versus distance and noise level, and to report where it fails.
- **Confounders:** coughs, sneezes, paper rustling, chairs, typing, air conditioning, invigilator speech, announcements, and exam-hall echo.
- **Languages and accents:** a signal-based detector is largely language-independent, but a learned one may not be. Evaluate across speakers.
- **Fairness and consent:** audio of people is personal data in most jurisdictions, and recording rules vary. Legal and ethical approval comes before collection.
- **A flag is a prompt for human review, not proof.**

## 4. Target Architecture (single laptop POC, ready for multi-node)

```
Microphone ──► Audio capture (16 kHz mono, 20-30 ms frames, timestamped)
                  │
                  ▼
           Audio features ──► Noise-floor calibration ──► Speech / whisper activity detector
                  │                                              │
                  ▼                                              ▼
        in-memory ring buffer                     Audio event session (duration, cooldown)
                                                                 │
Camera ► Pose (+ Face Mesh lips) ► Video session ───────────────┤
                                                                 ▼
                                             Fusion (audio risk × video context)
                                                                 │
                                                                 ▼
                          Risk score + alert ► Overlay, events.csv, optional evidence clip
                                                                 │
                                                  (later) compact message to controller
```

**Latency budget (to measure, not assume):** audio frame hop 20-30 ms; decision window 1-2 s of evidence; compute and fusion under 100 ms; alert emission under 2 s after the evidence threshold is met.

**New files (extends the module plan):**

```text
detector/
  audio_capture.py     # microphone source + fake source, shared clock
  audio_features.py    # frames -> band energies, flatness, zero-crossing, harmonicity
  audio_baseline.py    # ambient noise-floor calibration + slow adaptation
  audio_detector.py    # features -> {silence, speech, whisper, noise-burst}
  audio_session.py     # temporal logic for audio events
  lips.py              # Face Mesh lip-activity feature (video cue)
  fusion.py            # audio + video -> risk score and alerts
tools/
  record_session.py    # consented dataset recorder (labels + sync)
  eval_audio.py        # per-class metrics, distance/noise sweeps
  eval_fusion.py       # video-only vs audio-only vs fused ablation
```

## 5. New Requirements (add to the PRD)

| ID | Requirement |
| --- | --- |
| AUD-1 | Capture mono audio at 16 kHz from a selectable input device, with monotonic timestamps shared with the video clock |
| AUD-2 | Calibrate the room's noise floor on user command (`n`) and adapt it slowly, with the same safety gate used for the posture baseline |
| AUD-3 | Classify each 1-second window as silence/ambient, speech, whisper-like, or noise burst, with a confidence |
| AUD-4 | Raise an audio event only after sustained activity (default: whisper-like or speech for at least 3 s within a 10 s window), with a cooldown |
| AUD-5 | Compute lip-activity and head-orientation context from video; associate audio events with the most plausible candidate where possible |
| AUD-6 | Fuse audio and video into a risk score in 0..1 with a documented breakdown of contributing evidence |
| AUD-7 | Alert only when the fused risk exceeds a configurable threshold; audio-only events are logged at lower severity |
| AUD-8 | Never transcribe speech, never store voiceprints, never transmit raw audio |
| AUD-9 | Metadata-only mode by default; evidence clips only when explicitly enabled and consent is recorded |
| AUD-10 | Provide an `--no-audio` and a `--mute` switch (for announcements and invigilator speech) |
| AUD-11 | Record latency per stage and end-to-end event-to-alert latency |

## 6. Implementation Modules (vibe-coding format)

**Prerequisite:** the modular package from the earlier plan exists through at least M5 (config, features, classifier, session, baseline) with passing tests, or you do the refactor first. Use the same Master Rules prompt. **One new dependency needs your approval: `sounddevice` (PortAudio).** Everything else is NumPy and optionally SciPy.

Control commands are unchanged: `CONTINUE <ID>`, `STOP`, `STATUS`, `REVERT <ID>`, `RESCOPE`. Auto-halt conditions are unchanged, plus: **any code that records or transmits raw audio, or transcribes speech, triggers HALT.**

### SD0: Decisions, Ethics and Consent Gate (no code)

```text
STEP SD0: write docs/AUDIO_ETHICS.md. Cover: purpose limitation (detect acoustic
activity only), no transcription, no speaker identification, metadata-only default,
retention periods, consent text for mock-exam participants, a legal checklist for
audio recording in my jurisdiction (leave blanks I must verify), signage wording,
and an approval checklist that must be ticked before ANY human audio is recorded.
Do not write code. Stop after.
```

**Verify:** you read it, fill the blanks, and get the approvals your institution requires. **STOP if:** anything suggests recording without consent. **CONTINUE:** approvals are on file (or you are testing only with your own voice and synthetic audio) → `CONTINUE SD1`.

### SD1: Config and Audio Capture

```text
STEP SD1: add audio settings to config.py (sample_rate=16000, frame_ms=30,
window_s=1.0, device=None) and create detector/audio_capture.py with an AudioSource
protocol (read_frames(), close()), a MicSource using sounddevice with a callback that
puts timestamped frames into a bounded queue (drop-oldest, never block), and a
FakeAudioSource(array) plus a WAV-file source for replays. Provide a helper to list
input devices. Never write audio to disk. Tests: fake source yields correct frame count
and timestamps; queue overflow drops oldest; closing is idempotent. Ask me before adding
sounddevice to requirements. Report, STOP.
```

**Verify (manual):** list devices and print the RMS level while you speak. **CONTINUE:** → `SD2`.

### SD2: Audio Features (pure)

```text
STEP SD2: detector/audio_features.py, pure NumPy. For a frame or window compute: RMS
dB, band energies (100-300, 300-1000, 1000-4000, 4000-8000 Hz), spectral centroid,
spectral flatness, zero-crossing rate, and a harmonicity measure (normalized
autocorrelation peak in the 80-400 Hz pitch range). Guard silence and division by zero.
Tests with synthetic signals: sine at 150 Hz is highly harmonic; white noise has high
flatness and low harmonicity; band-limited noise (300-4000 Hz) peaks in speech bands;
silence returns floor values without NaN; results are stable if amplitude is scaled
(after dB normalization). Report, STOP.
```

**STOP if:** NaNs or infinities appear. **CONTINUE:** → `SD3`.

### SD3: Noise-Floor Calibration

```text
STEP SD3: detector/audio_baseline.py. NoiseFloor.calibrate(windows) stores mean and
std of dB and band energies from about 5 s of ambient sound. adapt(features, dt) updates
slowly (time constant 60 s) ONLY while the detector reports silence and the level is
within a gate of the current floor, so a sustained whisper is never learned as ambient.
Uncalibrated = conservative defaults. Tests: floor tracks slow HVAC changes; sustained
quiet speech is NOT absorbed; recalibration resets. Report, STOP.
```

**STOP if:** the sustained-speech test fails. **CONTINUE:** → `SD4`.

### SD4: Speech / Whisper Activity Detector (rules)

```text
STEP SD4: detector/audio_detector.py. classify_window(features, floor) -> (label,
confidence) with labels silence, speech, whisper, noise_burst. Heuristics, all thresholds
in config.py: activity = speech-band energy above floor by at least X dB;
speech = activity with high harmonicity and pitch in range; whisper = activity with low
harmonicity, high flatness, energy shifted to 1-8 kHz; noise_burst = very short,
broadband, high crest factor (cough, door, paper). Table-driven tests using synthetic
signals for each class plus near-miss cases (just under activity threshold -> silence).
Do not claim real-world accuracy; that comes from SD9. Report, STOP.
```

**CONTINUE:** → `SD5`.

### SD5: Audio Event Session (temporal logic)

```text
STEP SD5: detector/audio_session.py, pure, time passed as `now`. update(label,
confidence, now) -> list of AudioEvent(kind, duration, mean_conf, start, end). Rule: raise
an event when whisper/speech windows cover at least 3 s within a sliding 10 s window,
ignore noise_burst shorter than 1 s, cooldown 8 s, and honor a mute flag. Tests with
FakeClock (start at 1000.0): single cough does not raise; 3 s of whisper does; flicker
resets; cooldown works; mute suppresses. Report, STOP.
```

**CONTINUE:** → `SD6`.

### SD6: Video Context Cues (lips and orientation)

```text
STEP SD6: detector/lips.py using MediaPipe Face Mesh (mp.solutions.face_mesh, no extra
download) to compute normalized lip opening and its variation over 1 s (lip_activity in
0..1). Expose context() combining lip_activity with the existing pose cues: head turned
toward a neighbour, leaning, hand at mouth. Pure functions on landmarks plus a thin
wrapper class. Tests with synthetic landmarks: closed vs talking mouth; scale invariance;
missing face returns None. Check FPS cost and report it. Report, STOP.
```

**STOP if:** FPS drops below 15 on your laptop. Propose running face mesh every Nth frame and wait. **CONTINUE:** → `SD7`.

### SD7: Multimodal Fusion

```text
STEP SD7: detector/fusion.py. fuse(audio_events, video_context, now) -> list of
Alert(risk 0..1, severity, reasons[], evidence refs). Start with a transparent weighted
rule in config.py: risk = w_a*audio_conf + w_v*video_context + w_c*co-occurrence bonus
when audio activity overlaps a video cue (head turned toward neighbour or lip activity)
within the same 2 s window. Audio-only events become low-severity "review" items; fused
events above the threshold become alerts. Every alert lists its contributing evidence in
plain language. Tests: audio-only low severity; video-only unchanged; both together
exceed threshold; stale events expire; weights read from config. Report, STOP.
```

**CONTINUE:** → `SD8`.

### SD8: Overlay, Evidence and Privacy

```text
STEP SD8: extend overlay.py with an audio meter, noise-floor status, the current audio
label, and a risk bar with reasons. Extend evidence.py: add columns modality, risk,
reasons; keep ONE snapshot per flag; add an optional audio evidence clip (default OFF)
that saves the buffered 5 s before and 5 s after ONLY when --save-audio-clips is passed
AND a consent file exists. Add keys: n = calibrate noise floor, m = mute toggle.
Tests: metadata-only mode writes no audio files; clips require both the flag and the
consent file; mute suppresses audio events. PRIVACY TEST: no network calls and no audio
files in default mode. Report, STOP.
```

**CONTINUE:** → `SD9`.

### SD9: Consented Recording and Evaluation Tools

```text
STEP SD9: tools/record_session.py records synchronized timestamps and labels from a
scripted scenario (normal sitting, whisper at 1 m / 3 m, normal speech, cough, paper
rustle, typing, ventilation noise ON/OFF), requires an explicit consent confirmation at
start, and saves features and labels only (raw audio only if the participant opted in).
tools/eval_audio.py reports per-class precision, recall, F1, confusion matrix, false
alarms per hour, and detection rate vs distance and noise condition. tools/eval_fusion.py
runs video-only, audio-only and fused modes on the same sessions and prints an ablation
table. No new detection logic. Report, STOP.
```

**CONTINUE:** → `SD10`.

### SD10: Learned Classifier (optional, after data)

```text
STEP SD10: only after SD9 data exists. Train a small classifier (logistic regression or
a tiny CNN on log-mel features) using subject-disjoint train/validation/test splits.
Export to ONNX and add it behind the same classify_window() interface so rules and model
can be compared. Report per-class metrics against the rule baseline. Mark any claim not
backed by the held-out test set as UNTESTED. Report, STOP.
```

**STOP if:** the split is not subject-disjoint, since that leaks identity. **CONTINUE:** → `SD11`.

### SD11: Edge Hooks (multi-node readiness)

```text
STEP SD11: define a compact JSON alert schema (node_id, ts, risk, severity, reasons,
modality list, optional evidence ref) and a LocalPublisher interface with a console/file
implementation. No raw audio or video in messages; target under 1 KB per alert. Add
per-stage latency instrumentation and a benchmark that records end-to-end event-to-alert
latency (audio window excluded and reported separately). Prepare, but do not implement,
an MQTT/WebSocket publisher. Report, STOP.
```

**Done when:** the schema is documented and latency is measured on your hardware.

## 7. Test Strategy

| Level | What | How |
| --- | --- | --- |
| Unit | Features, detector, sessions, fusion | Synthetic signals (sine, white noise, band-limited noise, bursts) and a fake clock |
| Integration | Mic or WAV replay + camera or video replay | Fake sources, end-to-end run produces expected events |
| Privacy | No raw audio persisted or sent | File-system and network assertions in tests |
| Evaluation | Real-world accuracy | Consented mock sessions: distance (1, 3, 5 m), noise (silent, HVAC, typing), classes (whisper, speech, cough, rustle) |
| Ablation | Does audio help? | Video-only vs audio-only vs fused, same sessions |
| Performance | Latency, CPU, FPS | Per-stage timers; soak test 30 min |

## 8. Success Criteria for the Audio Extension

Following your document, numeric accuracy thresholds are set **after** baseline experiments. Fixed targets for now:

- Audio runs alongside video at 15 FPS or better on the target laptop.
- Event-to-alert latency at most 2 s after the evidence threshold is met (reported separately from the observation window).
- Zero audio-only alerts in a 5-minute quiet-room soak, and a reported false-alarm rate per hour under noisy conditions.
- A completed ablation table showing whether audio adds value over video alone.
- Privacy tests pass: no raw audio stored or transmitted by default; no transcription; no speaker identification.
- The system operates with no internet connection.

## 9. Timeline (fits Phases 5 and 6 of your schedule)

| Week | Work |
| --- | --- |
| 1 | SD0 ethics gate, SD1 capture, SD2 features |
| 2 | SD3 noise floor, SD4 detector, SD5 session |
| 3 | SD6 lip cues, SD7 fusion |
| 4 | SD8 overlay, evidence and privacy; SD11 hooks |
| 5-6 | SD9 consented recordings and evaluation (depends on ethics approval and participants) |
| After data | SD10 learned classifier |

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Whispers undetectable at distance with a laptop mic | High | Measure range honestly; evaluate a USB or array microphone; report limits |
| Many false alarms (coughs, rustling, HVAC) | High | Noise-floor calibration, burst rejection, duration rules, fusion requirement |
| Privacy and legal exposure from recording | High | SD0 gate, metadata-only default, no transcription, no voiceprints, consent file |
| Invigilator speech or announcements flagged | Medium | Mute toggle and an announcement time window; document the procedure |
| Face Mesh slows the pipeline | Medium | Run every Nth frame or only when audio activity is present |
| Learned model overfits to a few speakers | Medium | Subject-disjoint splits, more participants, report variance |
| Audio and video drift out of sync | Medium | Single monotonic clock, timestamp every chunk, tolerance window in fusion |
| Scope creep into speech recognition | Medium | Hard rule: HALT on any transcription code |

## 11. Open Questions for You

1. Which microphone will you use first: laptop built-in, a USB microphone, or a microphone array?
2. How far apart will candidates sit in the mock exam, and in which languages will participants whisper?
3. Do you have ethics approval and a consent form for audio, and may evidence clips be kept at all?
4. Should invigilator speech be excluded by a time window, a mute key, or both?
5. Is the first demo a single laptop (as now), or two nodes so the controller dashboard also receives audio-based alerts?
6. Do you want to compare the rule-based detector against a pretrained audio model (for example a small speech-activity model) as an additional baseline?

## 12. Suggested Next Step

Do **SD0** first, then SD1 to SD4 using synthetic and your own voice. That gives you a working audio meter and a rule-based talking/whisper detector within a few days, without waiting for ethics approval. The consented recordings in SD9 are the real test of whether whisper detection is viable in an exam room.