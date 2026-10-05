@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M9: Overlay renderer ===
echo.
%PY% -m pytest tests\test_overlay.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M9 M10 %RC%
exit /b %ERRORLEVEL%