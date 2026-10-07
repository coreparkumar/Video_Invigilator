@echo off
call "%~dp0_env.bat" || exit /b 1
echo === SD9: Real-world validation tools ===
echo.
echo Testing imports and syntax...
%PY% -m py_compile tools\audio_replay.py
if errorlevel 1 exit /b 1
%PY% -m py_compile tools\audio_meter.py
if errorlevel 1 exit /b 1
%PY% -m py_compile tools\audio_eval_sweep.py
if errorlevel 1 exit /b 1
echo Syntax OK.

:report
call "%~dp0_report.bat" SD9 SD10 0
exit /b 0