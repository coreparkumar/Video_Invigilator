@echo off
cd /d "%~dp0"
echo.
echo   M0  baseline freeze      M7   camera
echo   M1  config               M8   pose wrapper
echo   M2  features             M9   overlay
echo   M3  classifier           M10  app wiring
echo   M4  session              M11  evaluation
echo   M5  calibration          M12  docs and release
echo   M6  evidence logger
echo.
set /p N=Module number to test (0-12): 
if not exist m%N%_*.bat (
  echo No such module: %N%
  pause
  exit /b 1
)
for %%f in (m%N%_*.bat) do call "%%f"
exit /b %ERRORLEVEL%
