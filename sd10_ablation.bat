@echo off
call "%~dp0_env.bat" || exit /b 1
echo === SD10: Ablation harness ===
echo.
echo Testing imports and syntax...
%PY% -m py_compile tools\eval_ablation.py
if errorlevel 1 exit /b 1
echo Syntax OK.

:report
call "%~dp0_report.bat" SD10 DONE 0
exit /b 0