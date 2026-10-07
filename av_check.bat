@echo off
call "%~dp0_env.bat" || exit /b 1
echo === AV check: calibration quality + guided voice and video test ===
echo.
%PY% -m pytest tests -q --tb=short -k "calibration or avcheck or clipping"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" goto report
if defined AUTO goto report
if not exist tools\av_check.py goto report
echo.
choice /c YN /m "Run the live guided check now (camera + microphone)"
if errorlevel 2 goto report
%PY% tools\av_check.py
set RC=%ERRORLEVEL%
:report
call "%~dp0_report.bat" AVCHECK DONE %RC%
exit /b %ERRORLEVEL%
