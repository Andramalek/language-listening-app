@echo off
echo ===== Windows Camera Bridge for WSL =====
echo.
echo This script will start a camera server that allows WSL to access your webcam.
echo.

echo Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Python not found. Please install Python on Windows.
  echo Visit https://www.python.org/downloads/ to download Python.
  pause
  exit /b 1
)

echo Checking required packages...
python -c "import cv2" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Installing OpenCV...
  pip install opencv-python
)

python -c "import flask" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Installing Flask...
  pip install flask
)

echo.
echo Starting camera bridge...
echo Keep this window open while using the camera in WSL.
echo.
python windows_camera_bridge.py

pause 