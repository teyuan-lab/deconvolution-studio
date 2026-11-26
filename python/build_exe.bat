@echo off
rem ================================================================
rem  Builds dist\DeconvolutionStudio.exe  (a single file that runs
rem  on any 64-bit Windows PC without installing Python)
rem  Run this once on a PC that has Python. Takes about 3-5 minutes.
rem ================================================================
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (echo [ERROR] Python not found. & pause & exit /b 1)

echo [1/4] Creating a clean build environment (.build_venv) ...
if not exist ".build_venv\Scripts\python.exe" %PY% -m venv .build_venv
if errorlevel 1 (echo [ERROR] could not create the virtual environment & pause & exit /b 1)
set "VPY=.build_venv\Scripts\python.exe"

echo [2/4] Installing numpy, scipy, pillow, pyinstaller ...
"%VPY%" -m pip install --upgrade pip >nul
"%VPY%" -m pip install numpy scipy pillow pyinstaller
if errorlevel 1 (echo [ERROR] package installation failed - see above & pause & exit /b 1)

echo [3/4] Building DeconvolutionStudio.exe ...
"%VPY%" -m PyInstaller --noconfirm --clean DeconvolutionStudio.spec
if errorlevel 1 (echo [ERROR] PyInstaller failed - see above & pause & exit /b 1)

echo [4/4] Testing the exe ...
if exist "dist\selftest.txt" del "dist\selftest.txt"
start "" /wait "dist\DeconvolutionStudio.exe" --selftest
type "dist\selftest.txt"
findstr /c:"SELFTEST PASSED" "dist\selftest.txt" >nul 2>nul
if errorlevel 1 (echo [ERROR] the exe self-test failed - see above & pause & exit /b 1)

copy /y "dist\DeconvolutionStudio.exe" "DeconvolutionStudio.exe" >nul
powershell -NoProfile -Command "Compress-Archive -Force -Path 'DeconvolutionStudio.exe' -DestinationPath 'DeconvolutionStudio-Windows.zip'"

echo.
echo ================================================================
echo  Done.
echo    DeconvolutionStudio.exe          - double-click to run
echo    DeconvolutionStudio-Windows.zip  - the exe zipped, for e-mail
echo  No Python is needed on the PC that runs the exe.
echo ================================================================
pause
