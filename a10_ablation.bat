@echo off
call "%~dp0_env.bat" || exit /b 1
echo === A10: Ablation harness ===
echo.
%PY% -m pytest tests\test_ablation.py -q --tb=short
set RC=%ERRORLEVEL%
call "%~dp0_report.bat" A10 DONE %RC%
exit /b %ERRORLEVEL%
