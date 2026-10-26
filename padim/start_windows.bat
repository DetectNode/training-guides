@echo off
setlocal
cd /d "%~dp0"
title DetectNode PaDiM training
set PYTHONWARNINGS=ignore

where py >nul 2>nul
if not errorlevel 1 goto run_py
where python >nul 2>nul
if not errorlevel 1 goto run_python
where python3 >nul 2>nul
if not errorlevel 1 goto run_python3

echo.
echo Python 3 is not installed.
echo Install Python 3.10 or newer from https://www.python.org/downloads/
echo During installation, tick "Add python.exe to PATH".
echo.
start "" "https://www.python.org/downloads/"
goto end

:run_py
py -3 start.py
goto check

:run_python
python start.py
goto check

:run_python3
python3 start.py

:check
if errorlevel 1 (
  echo.
  echo Training failed. See the error above.
)

:end
echo.
pause
