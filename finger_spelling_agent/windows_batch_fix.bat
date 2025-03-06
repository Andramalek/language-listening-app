@echo off
cd /d "%~dp0"

echo ===== Windows Camera Bridge for WSL =====
echo.
echo This script will start a camera server that allows WSL to access your webcam.
echo.

REM Try different possible Python locations
set FOUND_PYTHON=0

REM Check Python in PATH
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found Python in PATH
    set PYTHON_CMD=python
    set FOUND_PYTHON=1
    goto :PYTHON_FOUND
)

REM Check Python launcher
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found Python launcher (py)
    set PYTHON_CMD=py
    set FOUND_PYTHON=1
    goto :PYTHON_FOUND
)

REM Check common Python installation locations
set PYTHON_LOCATIONS=^
C:\Python37\python.exe^
C:\Python38\python.exe^
C:\Python39\python.exe^
C:\Python310\python.exe^
C:\Python311\python.exe^
C:\Python312\python.exe^
C:\Program Files\Python37\python.exe^
C:\Program Files\Python38\python.exe^
C:\Program Files\Python39\python.exe^
C:\Program Files\Python310\python.exe^
C:\Program Files\Python311\python.exe^
C:\Program Files\Python312\python.exe^
C:\Program Files (x86)\Python37\python.exe^
C:\Program Files (x86)\Python38\python.exe^
C:\Program Files (x86)\Python39\python.exe^
C:\Program Files (x86)\Python310\python.exe^
C:\Program Files (x86)\Python311\python.exe^
C:\Program Files (x86)\Python312\python.exe^
%LOCALAPPDATA%\Programs\Python\Python37\python.exe^
%LOCALAPPDATA%\Programs\Python\Python38\python.exe^
%LOCALAPPDATA%\Programs\Python\Python39\python.exe^
%LOCALAPPDATA%\Programs\Python\Python310\python.exe^
%LOCALAPPDATA%\Programs\Python\Python311\python.exe^
%LOCALAPPDATA%\Programs\Python\Python312\python.exe

for %%p in (%PYTHON_LOCATIONS%) do (
    if exist "%%p" (
        echo Found Python at: %%p
        set PYTHON_CMD="%%p"
        set FOUND_PYTHON=1
        goto :PYTHON_FOUND
    )
)

:PYTHON_FOUND
if %FOUND_PYTHON% EQU 0 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version

REM Install required packages
echo.
echo Installing required packages...
%PYTHON_CMD% -m pip install opencv-python flask

REM Check if the camera bridge script exists
if not exist "%~dp0windows_camera_bridge.py" (
    echo ERROR: windows_camera_bridge.py not found!
    echo Please make sure it's in the same directory as this batch file.
    pause
    exit /b 1
)

REM Run the camera bridge
echo.
echo Starting camera bridge...
echo Keep this window open while using the camera in WSL.
echo.
%PYTHON_CMD% "%~dp0windows_camera_bridge.py"

pause 