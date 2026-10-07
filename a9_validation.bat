@echo off
call "%~dp0_env.bat" || exit /b 1
echo === A9: Real-world validation tools ===
echo.
%PY% -m pytest tests\test_audio_eval.py -q --tb=short
set RC=%ERRORLEVEL%
call "%~dp0_report.bat" A9 A10 %RC%
exit /b %ERRORLEVEL%
