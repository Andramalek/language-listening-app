"""
Manual Camera Test for ASL Finger Spelling Agent

This script allows you to manually test different camera sources,
which is especially useful when working with the Windows Camera Bridge in WSL.
"""

import cv2
import argparse
import sys
from utils.webcam_capture import WebcamCapture, test_webcam
from utils.webcam_utils import is_wsl

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Manual Camera Test")
    parser.add_argument(
        "--url", 
        type=str, 
        help="Camera URL to test (e.g., http://192.168.1.100:5000/video_feed)"
    )
    parser.add_argument(
        "--id", 
        type=int, 
        default=0, 
        help="Camera device ID to test (default: 0)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=10, 
        help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available camera devices"
    )
    args = parser.parse_args()
    
    # Print system info
    print("==== System Information ====")
    print(f"Running in WSL: {is_wsl()}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # List available cameras
    if args.list:
        list_cameras()
        return
    
    # Determine which camera source to use
    if args.url:
        print(f"\nTesting camera URL: {args.url}")
        source = args.url
    else:
        print(f"\nTesting camera ID: {args.id}")
        source = args.id
    
    # Run the test
    success = test_webcam(camera_source=source, duration=args.duration)
    
    if success:
        print("\n✅ Camera test successful!")
        print(f"Use this camera source in your application: {source}")
    else:
        print("\n❌ Camera test failed!")
        
        if args.url:
            print("\nTroubleshooting:")
            print("1. Make sure the Windows Camera Bridge is running")
            print("2. Check if the URL is correct")
            print("3. Try accessing the URL in a web browser")
            print("4. Check Windows Firewall settings")
        elif is_wsl():
            print("\nTroubleshooting in WSL:")
            print("1. Try using the Windows Camera Bridge instead")
            print("   Run: python manual_camera_test.py --url http://YOUR_WINDOWS_IP:5000/video_feed")
            print("2. Or enable native WSL camera support (see WEBCAM_SETUP.md)")

def list_cameras():
    """List available camera devices."""
    if is_wsl():
        print("\n⚠️ Running in WSL environment")
        print("Direct camera access may not work. Consider using the Windows Camera Bridge.")
        print("See WEBCAM_SETUP.md for instructions.")
    
    print("\nScanning for available camera devices...")
    
    # Try to find available cameras
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera info
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            available_cameras.append({
                "id": i,
                "resolution": f"{width}x{height}",
                "fps": fps
            })
            
            # Release the camera
            cap.release()
    
    if available_cameras:
        print(f"Found {len(available_cameras)} camera(s):")
        for cam in available_cameras:
            print(f"  Camera ID {cam['id']}: {cam['resolution']} @ {cam['fps']} FPS")
    else:
        print("No cameras found")
        
        if is_wsl():
            print("\nThis is expected in WSL. Please use the Windows Camera Bridge.")
            print("See WEBCAM_SETUP.md for instructions.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0) 