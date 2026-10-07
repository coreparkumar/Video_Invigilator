@echo off
rem Shared setup: go to project root, locate the venv python.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] No virtual environment found. Run build.bat first.
  if not defined NOPAUSE pause
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%"
exit /b 0
