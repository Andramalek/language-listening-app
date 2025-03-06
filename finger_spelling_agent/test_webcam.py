"""
Test Webcam Access in WSL

This script helps diagnose webcam access issues, particularly in WSL
environments where direct camera access can be challenging.
"""

import cv2
import time
import argparse
import sys
import numpy as np
import platform
from utils.webcam_utils import is_wsl, get_camera_source, get_webcam_instructions
from utils.webcam_capture import WebcamCapture
import streamlit as st

def test_direct_webcam(camera_id=0):
    """Test direct webcam access using OpenCV."""
    print(f"\n==== Testing direct OpenCV webcam access (camera_id={camera_id}) ====")
    
    # Try to open the camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera opened successfully")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("❌ Failed to read frame from camera")
        cap.release()
        return False
    
    print(f"✅ Successfully read frame of shape {frame.shape}")
    
    # Release resources
    cap.release()
    print("✅ Camera released")
    
    return True

def test_webcam_capture_class():
    """Test webcam access using our WebcamCapture class."""
    print(f"\n==== Testing WebcamCapture class ====")
    
    # Create webcam capture instance (auto-detects best source)
    webcam = WebcamCapture()
    
    # Try to start the webcam
    success = webcam.start()
    
    if not success:
        print("❌ Failed to start WebcamCapture")
        return False
    
    print("✅ WebcamCapture started successfully")
    
    # Try to get frames
    for i in range(3):
        print(f"   Reading frame {i+1}...")
        frame = webcam.get_frame()
        
        if frame is None:
            print(f"❌ Failed to get frame {i+1}")
        else:
            print(f"✅ Got frame {i+1} of shape {frame.shape}")
        
        time.sleep(0.5)
    
    # Stop webcam
    webcam.stop()
    print("✅ WebcamCapture stopped")
    
    return True

def print_system_info():
    """Print system information for debugging."""
    print("\n==== System Information ====")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Running in WSL: {is_wsl()}")
    
    # Check /dev/video* devices
    if platform.system() == "Linux":
        import glob
        video_devices = glob.glob("/dev/video*")
        print(f"Available video devices: {video_devices}")

def streamlit_test():
    """Streamlit interface for testing webcam."""
    st.title("Webcam Test in WSL")
    
    # System information
    st.header("System Information")
    st.write(f"Python version: {sys.version.split()[0]}")
    st.write(f"OpenCV version: {cv2.__version__}")
    st.write(f"Platform: {platform.platform()}")
    st.write(f"Running in WSL: {is_wsl()}")
    
    # Camera source
    camera_source = get_camera_source()
    st.write(f"Detected camera source: {camera_source}")
    
    # Test options
    st.header("Test Options")
    test_method = st.radio(
        "Select test method:",
        ["Direct OpenCV", "WebcamCapture Class", "Streamlit Native"]
    )
    
    if test_method == "Direct OpenCV":
        if st.button("Test Direct OpenCV Access"):
            result = test_direct_webcam(camera_source)
            if result:
                st.success("Direct OpenCV test successful!")
            else:
                st.error("Direct OpenCV test failed!")
                
                if is_wsl():
                    st.warning("WSL detected. See instructions below for webcam setup.")
                    st.markdown(get_webcam_instructions())
    
    elif test_method == "WebcamCapture Class":
        if st.button("Test WebcamCapture Class"):
            with st.spinner("Testing WebcamCapture class..."):
                webcam = WebcamCapture()
                if webcam.start():
                    frames = []
                    for i in range(3):
                        time.sleep(0.5)
                        frame = webcam.get_frame()
                        if frame is not None:
                            frames.append(frame)
                    
                    webcam.stop()
                    
                    if frames:
                        st.success(f"Successfully captured {len(frames)} frames!")
                        # Display the last frame
                        st.image(frames[-1], channels="BGR", caption="Last captured frame")
                    else:
                        st.error("Failed to capture any frames")
                else:
                    st.error("Failed to start WebcamCapture")
                    
                    if is_wsl():
                        st.warning("WSL detected. See instructions below for webcam setup.")
                        st.markdown(get_webcam_instructions())
    
    elif test_method == "Streamlit Native":
        st.write("Using Streamlit's built-in webcam functionality")
        img_file = st.camera_input("Take a picture")
        
        if img_file is not None:
            st.success("Successfully accessed webcam through Streamlit!")
            st.image(img_file)
    
    # WSL Instructions
    if is_wsl():
        st.header("WSL Webcam Setup")
        st.markdown(get_webcam_instructions())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test webcam access in WSL")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (default is Streamlit UI)")
    args = parser.parse_args()
    
    if args.cli:
        # CLI mode
        print_system_info()
        camera_source = get_camera_source()
        print(f"\nDetected camera source: {camera_source}")
        
        test_direct_webcam(camera_source)
        test_webcam_capture_class()
        
        if is_wsl():
            print("\n==== WSL Webcam Setup Instructions ====")
            print(get_webcam_instructions())
    else:
        # Streamlit UI mode
        streamlit_test() 