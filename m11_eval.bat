@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M11: Evaluation harness ===
echo.
echo Testing imports and syntax...
%PY% -m py_compile tools\eval_gestures.py
if errorlevel 1 exit /b 1
%PY% -m py_compile tools\soak_test.py
if errorlevel 1 exit /b 1
echo Syntax OK.

:report
call "%~dp0_report.bat" M11 M12 0
exit /b 0