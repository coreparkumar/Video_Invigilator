# Edge Invigilator: agent rules

Local, offline, real-time exam-room monitoring POC. Python 3.9-3.12, OpenCV, MediaPipe 0.10.14 (`mp.solutions`), NumPy, pytest. Windows or Linux.

## Commands
- Tests: `python -m pytest -q`  (one file: `python -m pytest tests/test_audio_x.py -q`)
- Synthetic audio self-test: `python tools/audio_selftest.py`
- Run video+audio app: `python gesture_poc_av.py` | video only: `python gesture_poc.py`
- Syntax check: `python -m py_compile <file>`

## Layout
`gesture_poc.py` (working video POC, do not modify) | `audio/` (config, capture, features, synth, baseline, detector, session, pipeline, fusion, lips, engine) | `tools/` | `tests/` | `docs/`

## Working rules (always)
1. One step at a time. Do ONLY the step I name, and edit ONLY the files in that step's scope.
2. Plan first: list files, tests and risks in at most 8 bullets, then wait for my "GO".
3. Tests first (or alongside), then code, then RUN them and show the real output. Never claim a test passed without running it.
4. Pure modules (features, baseline, detector, session, fusion, pipeline) have no camera, no file I/O, no system clock. Time is passed in as `now`.
5. Every threshold lives in `audio/config.py`. No magic numbers elsewhere.
6. No new dependency, network call, or `pip install` without asking me first.
7. Privacy: NO speech-to-text, NO speaker identification or voiceprints, NO raw audio written to disk, NO network. Only a CSV row and one snapshot per alert may be written. An audio-only result can never exceed low severity.
8. Honesty: never invent benchmarks, accuracy numbers or hardware results. Mark anything not run as UNTESTED. Say plainly when something could not be tested (camera, microphone).
9. Small typed functions, one-line docstrings, no clever abstractions.
10. `reference/` (if present) is an answer key I use for comparison. Never read, copy or import from it unless I explicitly ask.
11. When the step is complete print the CHECKPOINT REPORT below and STOP. Do not start the next step.

## CHECKPOINT REPORT
```
CHECKPOINT: <step id> <name>
Built:      <files + line counts>
Tests:      <command> -> <N passed / N failed>  (real output summary)
Diff:       <git diff --stat or file list>
Manual:     <exact command I should run + expected result>
Risks:      <anything uncertain, UNTESTED items>
Waiting for: next /aN command | /audio-halt | /undo | /audio-rescue
```

## Control words I may type
`STOP` (halt, no more edits, print state) | `GO` (approve the plan) | `/undo` (OpenCode built-in: revert the last change set)

## Auto-halt: print `HALT: <reason>` and stop when
a test fails twice after your fixes; a change needs a new dependency, a network call or an out-of-scope file; the diff is more than 50% over what you planned; the request is ambiguous; any code could save audio or transcribe speech.
