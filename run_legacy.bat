@echo off
call "%~dp0_env.bat" || exit /b 1
if exist gesture_poc.py (%PY% gesture_poc.py %*) else (%PY% legacy\gesture_poc.py %*)
exit /b %ERRORLEVEL%
