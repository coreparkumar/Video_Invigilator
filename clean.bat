@echo off
cd /d "%~dp0"
echo Removing caches...
for /d /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist demo.png del demo.png
if /i "%~1"=="all" (
  echo Also removing .venv and the evidence folder...
  if exist .venv rmdir /s /q .venv
  if exist evidence rmdir /s /q evidence
)
echo Done. Use "clean.bat all" to also remove .venv and evidence.
