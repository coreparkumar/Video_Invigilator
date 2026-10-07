@echo off
call "%~dp0_env.bat" || exit /b 1
if exist invigilator\__main__.py goto modular
echo [INFO] Modular package not built yet, running gesture_poc.py
if exist gesture_poc.py (%PY% gesture_poc.py %*) else (%PY% legacy\gesture_poc.py %*)
exit /b %ERRORLEVEL%
:modular
%PY% -m invigilator %*
exit /b %ERRORLEVEL%
