@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M6: Evidence logger ===
echo.
%PY% -m pytest tests\test_evidence.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M6 M7 %RC%
exit /b %ERRORLEVEL%