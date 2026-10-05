@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M10: App wiring and CLI ===
echo.
%PY% -m pytest tests\test_app.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M10 M11 %RC%
exit /b %ERRORLEVEL%