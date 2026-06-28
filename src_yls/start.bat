@echo off
setlocal enabledelayedexpansion

echo === YOLO ML Backend for Label Studio ===

REM --- Step 1: Set env vars from config.example.json ---
for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command ^
  "$c = Get-Content '%~dp0config.example.json' | ConvertFrom-Json; ^
   echo LABEL_STUDIO_URL=%%($c.LABEL_STUDIO_URL^); ^
   echo LABEL_STUDIO_API_KEY=%%($c.LABEL_STUDIO_API_KEY^)"`) do (
  set "%%a"
)
echo LABEL_STUDIO_URL=%LABEL_STUDIO_URL%
echo LABEL_STUDIO_API_KEY=*** (set)

REM --- Step 2: Override from config.json if exists ---
if exist "%~dp0config.json" (
    for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command ^
      "$c = Get-Content '%~dp0config.json' | ConvertFrom-Json; ^
       echo LABEL_STUDIO_URL=%%($c.LABEL_STUDIO_URL^); ^
       echo LABEL_STUDIO_API_KEY=%%($c.LABEL_STUDIO_API_KEY^)"`) do (
      set "%%a"
    )
    echo (overridden from config.json)
)

REM --- Step 3: Clean cache ---
echo Cleaning Label Studio cache...
set CACHE_DIR=%LOCALAPPDATA%\label-studio\Cache
if exist "!CACHE_DIR!" (
    del /q /s "!CACHE_DIR!\*" 2>nul
    echo Cache cleared.
) else (
    echo No cache dir found at !CACHE_DIR!
)

REM --- Step 4: Model check before server start ---
echo Checking model...
call conda run -n yolo26 python "%~dp0_wsgi.py" --port 9090 --host 0.0.0.0 --check
if %ERRORLEVEL% neq 0 (
    echo Model check FAILED. Server not started.
    pause
    exit /b 1
)

REM --- Step 5: Start server ---
echo Starting server on port 9090...
call conda run -n yolo26 python "%~dp0_wsgi.py" --port 9090 --host 0.0.0.0
pause
