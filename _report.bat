@echo off
rem Usage: _report.bat <module> <next-module|DONE> <exit-code>
echo.
if "%~3"=="0" goto pass
if "%~3"=="4" goto notbuilt
if "%~3"=="5" goto notbuilt
echo ============================================================
echo [FAIL] %~1  exit code %~3
echo Type STOP to the AI, paste the output above, and ask for a
echo root-cause analysis only. No edits until you approve.
echo ============================================================
goto done
:notbuilt
echo ============================================================
echo [NOT BUILT] %~1  test file missing or no tests collected.
echo Have you pasted this module's prompt and approved the work?
echo ============================================================
goto done
:pass
echo ============================================================
echo [PASS] %~1
if /i "%~2"=="DONE" goto finished
echo Review the AI checkpoint report, then type: CONTINUE %~2
goto bar
:finished
echo All modules complete. Tag the release: git tag poc-v1.0
:bar
echo ============================================================
:done
if not defined NOPAUSE pause
exit /b %~3
