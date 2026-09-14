@echo off
rem H3 Shorts Studio v2 - double-click launcher (ASCII only).
rem Uses the v2 app server (FAST default). v1 runners are untouched.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\RUN_H3_APP.ps1"
if errorlevel 1 (
  echo.
  echo [H3-V2] Startup failed with exit code %errorlevel%.
  echo See the messages above. The window stays open.
  pause
  exit /b %errorlevel%
)
