@echo off
setlocal
cd /d "%~dp0"
echo === Edge Invigilator: build ===
set "PYCMD="
for %%v in (3.12 3.11 3.10 3.9) do if not defined PYCMD py -%%v -c "import sys" >nul 2>&1 && set "PYCMD=py -%%v"
if not defined PYCMD python -c "import sys; sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "PYCMD=python"
if not defined PYCMD goto nopython
echo Using: %PYCMD%

ver >nul
if not exist .venv\Scripts\python.exe %PYCMD% -m venv .venv
if errorlevel 1 goto fail
set "PY=.venv\Scripts\python.exe"

echo Installing dependencies...
%PY% -m pip install --upgrade pip -q
%PY% -m pip install -r requirements.txt pytest -q
if errorlevel 1 goto fail

echo Verifying imports...
%PY% -c "import cv2, mediapipe as mp; assert hasattr(mp,'solutions'), 'wrong mediapipe version'; print('cv2', cv2.__version__, '| mediapipe', mp.__version__)"
if errorlevel 1 goto fail

echo Syntax check...
if exist gesture_poc.py %PY% -m py_compile gesture_poc.py
if errorlevel 1 goto fail
if exist invigilator %PY% -m compileall -q invigilator
if errorlevel 1 goto fail
if exist tools %PY% -m compileall -q tools
if errorlevel 1 goto fail

echo.
echo [OK] Build complete.
echo   run_legacy.bat   run the current single-file POC
echo   steps.bat        test one module
echo   status.bat       pass/fail table for M0 to M10
if not defined NOPAUSE pause
exit /b 0

:nopython
echo [ERROR] Python 3.9 to 3.12 not found. MediaPipe 0.10.14 does not support newer versions.
echo Install Python 3.12 from python.org, then re-run build.bat.
if not defined NOPAUSE pause
exit /b 1

:fail
echo.
echo [FAIL] Build failed. Check the messages above.
if not defined NOPAUSE pause
exit /b 1
