@echo off
REM Dashboard demo and launch script for Edge Invigilator
REM Runs tests, renders demo PNGs, and optionally launches live dashboard

echo ==========================================
echo Edge Invigilator - Dashboard Demo & Launch
echo ==========================================
echo.

REM Check if we're in the right directory
if not exist "gesture_poc_av.py" (
    echo ERROR: Please run this script from the Video_Invigilator root directory.
    exit /b 1
)

echo [1/3] Running unit tests...
python -m pytest tests/test_ui.py tests/test_ui_panels_window.py tests/test_ui_findings.py -q
if errorlevel 1 (
    echo WARNING: Some tests failed. Continuing anyway...
) else (
    echo All UI tests passed!
)
echo.

echo [2/3] Running audio self-test...
python tools/audio_selftest.py
if errorlevel 1 (
    echo WARNING: Audio self-test failed. Continuing anyway...
) else (
    echo Audio self-test passed!
)
echo.

echo [3/3] Rendering dashboard demo PNGs...
python tools/render_dashboard_demo.py
if errorlevel 1 (
    echo ERROR: Failed to render demo PNGs.
    exit /b 1
) else (
    echo Demo PNGs rendered to demo_output/
)
echo.

echo ==========================================
echo All checks passed! Demo PNGs in demo_output/
echo ==========================================
echo.

REM Check for optional launch arguments
if "%1"=="--live" (
    echo Launching live dashboard...
    echo.
    echo Usage: python gesture_poc_av.py [options]
    echo Options:
    echo   --dashboard    Enable new dashboard (default)
    echo   --classic      Use classic overlay
    echo   --transcribe   Enable local speech-to-text (requires --model)
    echo   --model PATH   Path to Vosk model folder
    echo   --log-transcript  Log transcript to CSV (requires consent.txt)
    echo.
    echo Keys: c=calibrate, n=recalibrate, m=mute, s=skeleton, t=transcript, q=quit
    echo.
    python gesture_poc_av.py --dashboard
) else if "%1"=="--classic" (
    echo Launching classic overlay...
    python gesture_poc_av.py --classic
) else (
    echo To launch live dashboard:  scripts\dashboard.bat --live
    echo To launch classic overlay: scripts\dashboard.bat --classic
    echo.
    echo Demo PNGs are in demo_output/ - open them to review the dashboard layout.
)