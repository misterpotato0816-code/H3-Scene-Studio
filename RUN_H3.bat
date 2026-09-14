@echo off
rem H3 launcher (ASCII only). Thin wrapper around: python -X utf8 -m h3app.launcher start
rem Never requests admin elevation. All paths are quoted so spaces and
rem non-ASCII characters in the install path work.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "APPDIR=%~dp0app"
set "CFGFILE=%APPDIR%\config.json"
set "PYEXE="
set "PYARG="

rem 1) Try the ComfyUI python recorded in app\config.json ("comfy_python").
rem NOTE: tokens=1* keeps everything after the FIRST colon, so Windows
rem drive letters (E:\...) in the value survive the split.
if exist "%CFGFILE%" (
  for /f "usebackq tokens=1* delims=:" %%A in (`findstr /c:"\"comfy_python\"" "%CFGFILE%"`) do (
    set "LINE=%%B"
    set "LINE=!LINE:,=!"
    set "LINE=!LINE:"=!"
    for /f "tokens=* delims= " %%B in ("!LINE!") do set "LINE=%%B"
    if not "!LINE!"=="" if exist "!LINE!" set "PYEXE=!LINE!"
  )
)

rem 2) Fall back to <comfy_dir>\.venv\Scripts\python.exe from the same file.
if not defined PYEXE if exist "%CFGFILE%" (
  for /f "usebackq tokens=1* delims=:" %%A in (`findstr /c:"\"comfy_dir\"" "%CFGFILE%"`) do (
    set "LINE=%%B"
    set "LINE=!LINE:,=!"
    set "LINE=!LINE:"=!"
    for /f "tokens=* delims= " %%B in ("!LINE!") do set "LINE=%%B"
    if not "!LINE!"=="" if exist "!LINE!\.venv\Scripts\python.exe" set "PYEXE=!LINE!\.venv\Scripts\python.exe"
  )
)

rem 3) Fall back to whatever python the system has on PATH.
if not defined PYEXE (
  where py >nul 2>nul && (set "PYEXE=py" & set "PYARG=-3")
)
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  echo Python not found. Check the ComfyUI .venv or install Python on PATH.
  pause
  exit /b 1
)

pushd "%APPDIR%"
"%PYEXE%" %PYARG% -X utf8 -m h3app.launcher start
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
  echo.
  echo H3 launcher failed with exit code %RC%.
  pause
)
exit /b %RC%
