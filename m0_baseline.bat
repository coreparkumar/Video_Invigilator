@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M0: Baseline freeze and scaffold ===
echo.
%PY% -m pytest tests\test_characterization.py -q --tb=short
set RC=%ERRORLEVEL%

set "SCAF=0"
if not exist invigilator\__init__.py set "SCAF=1" & echo [WARN] invigilator package folder missing
if not exist tests\conftest.py set "SCAF=1" & echo [WARN] tests\conftest.py missing
if not exist legacy\gesture_poc.py if not exist gesture_poc.py set "SCAF=1" & echo [WARN] legacy POC file not found
if "%RC%"=="0" if "%SCAF%"=="1" set RC=1

:report
call "%~dp0_report.bat" M0 M1 %RC%
exit /b %ERRORLEVEL%
