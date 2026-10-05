@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M7: Camera source ===
echo.
%PY% -m pytest tests\test_camera.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M7 M8 %RC%
exit /b %ERRORLEVEL%