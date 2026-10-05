@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M8: Pose estimator wrapper ===
echo.
%PY% -m pytest tests\test_pose.py -q --tb=short
set RC=%ERRORLEVEL%

:report
call "%~dp0_report.bat" M8 M9 %RC%
exit /b %ERRORLEVEL%