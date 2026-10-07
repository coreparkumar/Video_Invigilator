@echo off
call "%~dp0_env.bat" || exit /b 1
%PY% -m pytest -q --tb=short %*
set RC=%ERRORLEVEL%
if not defined NOPAUSE pause
exit /b %RC%
