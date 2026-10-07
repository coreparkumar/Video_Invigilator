@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M1: Configuration ===
echo.
%PY% -m pytest tests\test_config.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M1 M2 %RC%
exit /b %ERRORLEVEL%
