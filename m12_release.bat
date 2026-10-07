@echo off
call "%~dp0_env.bat" || exit /b 1
echo === M12: Docs and release ===
echo.
%PY% -m pytest -q --tb=short
set RC=%ERRORLEVEL%
if not "%RC%"=="0" goto report
if defined AUTO goto report
echo.
echo --- Fresh-environment install check (takes a few minutes) ---
if exist .venv_fresh rmdir /s /q .venv_fresh
%PY% -m venv .venv_fresh
if errorlevel 1 goto freshfail
.venv_fresh\Scripts\python.exe -m pip install -q -r requirements.txt pytest
if errorlevel 1 goto freshfail
.venv_fresh\Scripts\python.exe -m pytest -q --tb=short
set RC=%ERRORLEVEL%
goto cleanup
:freshfail
set RC=1
:cleanup
if exist .venv_fresh rmdir /s /q .venv_fresh
:report
call "%~dp0_report.bat" M12 DONE %RC%
exit /b %ERRORLEVEL%
