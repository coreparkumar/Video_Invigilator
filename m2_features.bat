@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M2: Feature extraction ===
echo.
%PY% -m pytest tests\test_features.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M2 M3 %RC%
exit /b %ERRORLEVEL%
