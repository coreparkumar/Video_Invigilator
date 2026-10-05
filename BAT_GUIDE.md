# Batch Files Guide (Windows)

Double-click or run from a terminal in this folder.

| File | Purpose |
|---|---|
| `build.bat` | Finds Python 3.9-3.12, creates `.venv`, installs pinned deps + pytest, verifies imports, syntax-checks code |
| `run.bat` | Runs the modular app once M10 exists, otherwise the legacy `gesture_poc.py` |
| `run_legacy.bat` | Always runs the single-file POC |
| `steps.bat` | Menu: pick a module number (0-12) and test it |
| `status.bat` | Pass / fail / not-built table for M0-M10 (skips manual checks) |
| `test_all.bat` | Full pytest run (extra args pass through, e.g. `test_all.bat -k session`) |
| `clean.bat` | Removes caches; `clean.bat all` also removes `.venv` and `evidence` |
| `scripts\m0_...` to `m12_...` | One test script per implementation step |

## Per-step flow
1. AI builds module Mx and prints its checkpoint report.
2. You run `scripts\mX_*.bat` (or `steps.bat`).
3. `[PASS]` prints the exact continue command, for example `CONTINUE M4`.
   `[FAIL]` tells you to type `STOP` and paste the output to the AI.
   `[NOT BUILT]` means the module's tests don't exist yet.

## Manual checks (skipped when `AUTO=1`, as in `status.bat`)
- M7 camera discovery, M8 pose smoke test, M9 renders `demo.png`, M10 offers to launch the app.
- M11 runs the guided gesture test and a 2-minute soak test (needs you in front of the camera).
- M12 also performs a fresh-venv install and test run.

Exit codes: `0` pass, `4` or `5` not built (pytest found no tests), anything else fail.
