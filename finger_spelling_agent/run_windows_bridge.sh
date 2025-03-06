#!/bin/bash
# Helper script to run Windows Camera Bridge from WSL

# Get Windows username
WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')

# Path to the batch file
BATCH_FILE="/mnt/c/Users/$WIN_USER/Desktop/run_camera_bridge.bat"
WIN_PATH=$(wslpath -w "$BATCH_FILE")

echo "===== Running Windows Camera Bridge from WSL ====="
echo "Windows Username: $WIN_USER"
echo "Batch file path: $BATCH_FILE"
echo "Windows path: $WIN_PATH"
echo

# Check if the file exists
if [ ! -f "$BATCH_FILE" ]; then
    echo "ERROR: Batch file not found at $BATCH_FILE"
    echo "Please make sure run_camera_bridge.bat is on your Windows Desktop"
    exit 1
fi

echo "Starting Windows Camera Bridge..."
echo "A Command Prompt window should open on Windows."
echo

# Run the batch file
cmd.exe /c "$WIN_PATH"

echo
echo "If the Command Prompt window opened successfully,"
echo "you should now be able to access the camera in WSL."
echo
echo "Try running:"
echo "python manual_camera_test.py --url \"http://192.168.0.32:8080/video_feed\"" 