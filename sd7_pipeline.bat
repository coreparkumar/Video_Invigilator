@echo off
call "%~dp0_env.bat" || exit /b 1
echo === SD7: Streaming pipeline and synthetic self-test ===
echo.
%PY% -m pytest tests\test_audio_pipeline.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" SD7 SD8 %RC%
exit /b %ERRORLEVEL%