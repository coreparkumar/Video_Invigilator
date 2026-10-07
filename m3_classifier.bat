@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M3: Gesture classifier ===
echo.
%PY% -m pytest tests\test_classifier.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M3 M4 %RC%
exit /b %ERRORLEVEL%