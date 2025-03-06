# Windows Camera Bridge for WSL

This guide provides detailed instructions for setting up webcam access in WSL using the Windows Camera Bridge method.

## What is the Windows Camera Bridge?

The Windows Camera Bridge is a simple Python server that runs on Windows and makes your webcam available to WSL through HTTP streaming. This is necessary because WSL doesn't have direct access to hardware devices like webcams.

## Step-by-Step Setup Instructions

### Step 1: Copy Files to Windows

First, you need to copy the necessary files to your Windows file system:

1. Navigate to the `finger_spelling_agent` directory in WSL
2. Copy these files to your Windows filesystem (e.g., Desktop or Documents folder):
   - `windows_camera_bridge.py` - The main camera bridge script
   - `run_camera_bridge.bat` - Helper batch file to run the script on Windows

You can copy the files using the WSL file explorer, or using commands like:

```bash
cp windows_camera_bridge.py /mnt/c/Users/YourUsername/Desktop/
cp run_camera_bridge.bat /mnt/c/Users/YourUsername/Desktop/
```

Replace `YourUsername` with your actual Windows username.

### Step 2: Run the Camera Bridge on Windows

1. Open File Explorer on Windows
2. Navigate to where you copied the files (e.g., Desktop)
3. Double-click `run_camera_bridge.bat`

This batch file will:
- Check if Python is installed on Windows
- Install necessary packages (OpenCV and Flask) if needed
- Start the camera bridge server

You should see output similar to:

```
===== Windows Camera Bridge for WSL =====

This script will start a camera server that allows WSL to access your webcam.

Checking Python installation...
Checking required packages...

Starting camera bridge...
Keep this window open while using the camera in WSL.

Camera opened successfully!
Resolution: 640.0x480.0
FPS: 30.0
Starting camera server...
Access the camera in WSL via http://localhost:5000/video_feed
Local IP address: 192.168.1.100
Try http://192.168.1.100:5000/video_feed from WSL if localhost doesn't work
 * Serving Flask app 'windows_camera_bridge'
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.100:5000
```

**Important**: Keep this window open while using the camera in WSL.

### Step 3: Test the Connection from WSL

Now that the Windows Camera Bridge is running, let's test the connection from WSL:

1. Open your WSL terminal
2. Navigate to the finger_spelling_agent directory:
   ```
   cd ~/projects/clone/language_listening_app/finger_spelling_agent
   ```
3. Activate the virtual environment:
   ```
   source ../venv/bin/activate
   ```
4. Run the test script:
   ```
   python test_direct_bridge.py
   ```

If the connection test fails with the default URL (localhost:5000), the script will automatically try alternative URLs. If a working URL is found, you'll see output like:

```
📌 Found working camera bridge URL: http://192.168.1.100:5000/video_feed
📌 Use this URL in your application!
```

Make a note of this URL - you'll need it for the next step.

### Step 4: Run the ASL Finger Spelling Agent

Once you have a working camera bridge URL, you can run the ASL Finger Spelling Agent:

1. Navigate to the finger_spelling_agent directory:
   ```
   cd ~/projects/clone/language_listening_app/finger_spelling_agent
   ```
2. Activate the virtual environment:
   ```
   source ../venv/bin/activate
   ```
3. Run the streamlit app:
   ```
   streamlit run app.py
   ```
4. Access the app in your browser at the URL shown in the terminal (usually http://localhost:8501)

The ASL Finger Spelling Agent should automatically detect the Windows camera bridge and use it for webcam access.

## Troubleshooting

### Can't Connect to the Camera Bridge

If WSL can't connect to the Windows camera bridge, try:

1. **Check Firewall Settings**: Make sure Windows Firewall isn't blocking the connection
   - Open Windows Defender Firewall
   - Click "Allow an app or feature through Windows Defender Firewall"
   - Click "Change settings"
   - Click "Allow another app..."
   - Browse to Python executable (usually in `C:\Users\YourUsername\AppData\Local\Programs\Python\`)
   - Make sure both Private and Public networks are checked

2. **Use the IP Address**: Instead of `localhost`, use the Windows IP address
   - The camera bridge prints this as "Local IP address" when starting
   - Try using that address in WSL: `http://192.168.1.100:5000/video_feed` (your IP will be different)

3. **Check WSL Network**: Make sure WSL can reach the Windows host
   - Try pinging the Windows host from WSL: `ping 192.168.1.100` (replace with your Windows IP)

### Camera Not Working on Windows

If the camera doesn't work on Windows:

1. Check if other applications can access the camera
2. Make sure the camera isn't already in use by another application
3. Try a different camera ID in the `windows_camera_bridge.py` script (change the line `camera = cv2.VideoCapture(0)` to use a different number)

## Advanced Configuration

### Using a Specific Camera

If you have multiple cameras connected to your Windows machine, you can specify which one to use:

1. Open `windows_camera_bridge.py` in a text editor
2. Find this line: `camera = cv2.VideoCapture(0)`
3. Change the `0` to another number (1, 2, etc.) to select a different camera

### Changing Resolution or FPS

To change the camera resolution or frame rate:

1. Open `windows_camera_bridge.py` in a text editor
2. Find these lines:
   ```python
   camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   camera.set(cv2.CAP_PROP_FPS, 30)
   ```
3. Change the values as needed (e.g., change 640x480 to 1280x720 for higher resolution) 