@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M4: Temporal session logic ===
echo.
%PY% -m pytest tests\test_session.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M4 M5 %RC%
exit /b %ERRORLEVEL%