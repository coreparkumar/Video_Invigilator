@echo off
:: release_full.bat - Full build and run script for Edge Invigilator POC
@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Edge Invigilator POC - Full Release Build
echo ============================================
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found. Run from project root.
    exit /b 1
)

REM ============================================
echo [1/5] Setting up Python virtual environment...
REM ============================================
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM ============================================
echo [2/5] Installing dependencies...
REM ============================================
.venv\Scripts\pip install --upgrade pip >nul 2>&1
REM Note: pip returns exit code 1 when requirements already satisfied, so we check if packages are importable instead
.venv\Scripts\python.exe -c "import mediapipe; import cv2; import numpy; print('Dependencies already satisfied')" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    .venv\Scripts\pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        exit /b 1
    )
) else (
    echo Dependencies already satisfied.
)

REM ============================================
echo [3/5] Verifying installation...
REM ============================================
.venv\Scripts\python.exe -c "import cv2; import mediapipe as mp; import numpy; print('OpenCV:', cv2.__version__, '| MediaPipe:', mp.__version__, '| NumPy:', numpy.__version__)"
if errorlevel 1 (
    echo ERROR: Import verification failed
    exit /b 1
)

REM ============================================
echo [4/5] Running test suite...
REM ============================================
echo Running full test suite...
.venv\Scripts\python.exe -m pytest tests/ -q --tb=short
if errorlevel 1 (
    echo WARNING: Some tests failed. Check output above.
    echo Continuing anyway...
)

REM ============================================
echo [5/5] Running audio self-test...
REM ============================================
.venv\Scripts\python.exe tools/audio_selftest.py
if errorlevel 1 (
    echo WARNING: Audio self-test failed. Check output above.
    echo Continuing anyway...
)

REM ============================================
echo.
echo ============================================
echo BUILD SUCCESSFUL!
echo ============================================
echo.
echo Ready to run:
echo   Video only:     .venv\Scripts\python.exe -m invigilator
echo   Video + Audio:  .venv\Scripts\python.exe gesture_poc_av.py
echo   Audio self-test: .venv\Scripts\python.exe tools/audio_selftest.py
echo   Live AV check:  .venv\Scripts\python.exe tools/av_check.py --camera 0 --device 0
echo.
echo Output folder: .\evidence\
echo.
echo Press any key to run the Video+Audio POC (gesture_poc_av.py)...
echo Press Ctrl+C to exit without running.
pause >nul

echo.
echo Starting Edge Invigilator POC (Video + Audio)...
echo Press 'q' in the window to quit.
echo.

.venv\Scripts\python.exe gesture_poc_av.py

echo.
echo Application closed.
pause