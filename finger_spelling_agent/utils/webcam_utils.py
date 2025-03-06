"""
Webcam utilities for the ASL Finger Spelling Agent.
Provides functions to handle webcam access in different environments,
including WSL where direct camera access might be challenging.
"""

import os
import sys
import cv2
import platform
import subprocess
import socket

def is_wsl():
    """Check if running in Windows Subsystem for Linux (WSL)."""
    if platform.system() == "Linux":
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return True
        except:
            pass
    return False

def is_windows_bridge_running(host="localhost", port=5000):
    """Check if the Windows camera bridge is running."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False

def get_camera_source():
    """
    Determine the best camera source based on the environment.
    
    Returns:
        str or int: Camera source identifier
            - Integer (e.g., 0) for direct device access
            - URL string for HTTP streaming in WSL
    """
    # If running in WSL, try to use the Windows bridge
    if is_wsl():
        print("Detected WSL environment.")
        
        # Check if the Windows bridge is running
        if is_windows_bridge_running():
            print("Windows camera bridge detected at http://localhost:5000")
            return "http://localhost:5000/video_feed"
        else:
            print("Windows camera bridge not detected.")
            print("Please run windows_camera_bridge.py on your Windows host:")
            print("1. Open a Command Prompt or PowerShell window in Windows")
            print("2. Navigate to this directory")
            print("3. Run: python windows_camera_bridge.py")
            print("\nUsing default camera source, but this might not work in WSL...")
    
    # Default to camera ID 0 (primary camera)
    return 0

def get_webcam_instructions():
    """Return instructions for setting up webcam access in WSL."""
    if not is_wsl():
        return None
        
    return """
    ## WSL Webcam Setup Instructions
    
    To use your webcam in WSL, follow these steps:
    
    ### Option 1: Use the Windows Camera Bridge (Recommended)
    
    1. Copy the `windows_camera_bridge.py` script to your Windows filesystem
    2. Open Command Prompt or PowerShell in Windows
    3. Install the required packages:
       ```
       pip install opencv-python flask
       ```
    4. Run the bridge script:
       ```
       python windows_camera_bridge.py
       ```
    5. The camera feed will be available at http://localhost:5000/video_feed
    
    ### Option 2: Enable Native WSL Camera Support
    
    1. Open Command Prompt or PowerShell as Administrator
    2. Run:
       ```
       notepad %USERPROFILE%\.wslconfig
       ```
    3. Add the following lines:
       ```
       [wsl2]
       cameras=on
       ```
    4. Save and close the file
    5. Restart WSL:
       ```
       wsl --shutdown
       ```
    6. Reopen your WSL terminal and try again
    """ 