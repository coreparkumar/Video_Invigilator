@echo off
cd /d "%~dp0"
set NOPAUSE=1
set AUTO=1
echo.
echo === Automated status M0 to M10 (manual checks skipped) ===
for %%f in (m?_*.bat m10_*.bat) do (
  call "%%f" >nul 2>&1
  if errorlevel 6 (echo %%~nf : FAIL) else if errorlevel 4 (echo %%~nf : NOT BUILT) else if errorlevel 1 (echo %%~nf : FAIL) else (echo %%~nf : PASS)
)
echo.
echo For details run steps.bat and pick the module.
pause
