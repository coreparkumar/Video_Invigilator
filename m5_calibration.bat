@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M5: Calibration and adaptive baseline ===
echo.
%PY% -m pytest tests\test_baseline.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M5 M6 %RC%
exit /b %ERRORLEVEL%