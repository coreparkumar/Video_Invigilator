@echo off
call "%~dp0_env.bat" || exit /b 1
echo === SD8: Engine and live integration ===
echo.
%PY% -m pytest tests\test_audio_engine.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" SD8 SD9 %RC%
exit /b %ERRORLEVEL%