@echo off
rem ===== Deconvolution Studio launcher =====
cd /d "%~dp0"

rem --- find Python (prefer the py launcher, then python on PATH) ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [ERROR] Python was not found. Install it from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" during setup.
  pause
  exit /b 1
)

rem --- check required packages, install them on first run ---
%PY% -c "import numpy, scipy, matplotlib, PIL, tkinter" >nul 2>nul
if errorlevel 1 (
  echo First run: installing numpy, scipy, matplotlib ...
  %PY% -m pip install -r requirements.txt
  %PY% -c "import numpy, scipy, matplotlib, PIL, tkinter"
  if errorlevel 1 (
    echo.
    echo [ERROR] Required packages could not be loaded - see the message above.
    echo         If it says "No module named tkinter", re-run the Python installer,
    echo         choose Modify and enable "tcl/tk and IDLE".
    pause
    exit /b 1
  )
)

rem --- start the app ---
echo Starting Deconvolution Studio ...  (this window can be minimised)
%PY% app.py %*
if errorlevel 1 (
  echo.
  echo [ERROR] The app stopped with an error. Details are above and in startup_error.log
  pause
)
