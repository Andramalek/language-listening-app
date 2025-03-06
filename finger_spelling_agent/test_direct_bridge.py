"""
Direct Test for Windows Camera Bridge

This script tries to directly access the Windows camera bridge at http://localhost:8080/video_feed
without relying on automatic detection.
"""

import cv2
import numpy as np
import time

def test_bridge_connection():
    """Test if we can connect to the bridge server."""
    import socket
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex(('localhost', 8080))
        s.close()
        
        if result == 0:
            print("✅ Successfully connected to camera bridge server at localhost:8080")
            return True
        else:
            print(f"❌ Failed to connect to camera bridge server: error code {result}")
            return False
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def test_direct_bridge_access():
    """Test direct access to the Windows camera bridge."""
    print("\n==== Testing direct access to Windows camera bridge ====")
    
    # Test connection first
    if not test_bridge_connection():
        print("⚠️ Cannot connect to the camera bridge server")
        print("Make sure the Windows camera bridge script is running on Windows")
        return False
    
    # Try to open the video stream
    print("Trying to open video stream...")
    
    # Direct URL to the camera feed
    url = "http://localhost:8080/video_feed"
    cap = cv2.VideoCapture(url)
    
    # Wait a bit for connection to establish
    time.sleep(2)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video stream at {url}")
        return False
    
    print(f"✅ Successfully opened video stream")
    
    # Try to read some frames
    frames_read = 0
    start_time = time.time()
    timeout = 5  # seconds
    
    while frames_read < 10 and (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frames_read += 1
            print(f"   Frame {frames_read}: shape={frame.shape}")
        else:
            print("   Failed to read frame")
        
        time.sleep(0.2)
    
    # Release the capture
    cap.release()
    
    if frames_read > 0:
        print(f"✅ Successfully read {frames_read} frames from the camera bridge")
        return True
    else:
        print("❌ Failed to read any frames from the camera bridge")
        return False

def try_alternative_url():
    """Try an alternative URL for the Windows camera bridge."""
    # Try an IP address instead of localhost
    import socket
    
    print("\n==== Trying alternative URLs for camera bridge ====")
    
    # Get the WSL IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't have to be reachable
        s.connect(('8.8.8.8', 1))
        wsl_ip = s.getsockname()[0]
    except Exception:
        wsl_ip = '127.0.0.1'
    finally:
        s.close()
    
    # Generate possible Windows host IPs based on the WSL IP
    possible_ips = []
    
    # Common Windows host IP when using WSL2
    if wsl_ip.startswith('172.'):
        parts = wsl_ip.split('.')
        possible_ips.append(f"172.{parts[1]}.{parts[2]}.1")
    
    # Add common WSL host possibilities
    possible_ips.extend(['172.17.0.1', '172.18.0.1', '172.19.0.1', '172.20.0.1', '172.21.0.1'])
    possible_ips.extend(['192.168.0.1', '192.168.1.1', '10.0.0.1', '10.0.0.2'])
    
    # Add local IP
    possible_ips.append('127.0.0.1')
    
    # Get the computer hostname
    hostname = socket.gethostname()
    print(f"WSL IP address: {wsl_ip}")
    print(f"Computer hostname: {hostname}")
    
    # Try each possible IP
    for ip in possible_ips:
        url = f"http://{ip}:8080/video_feed"
        print(f"\nTrying URL: {url}")
        
        # Test connection
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex((ip, 8080))
            s.close()
            
            if result == 0:
                print(f"✅ Successfully connected to {ip}:8080")
                
                # Try to open video stream
                cap = cv2.VideoCapture(url)
                time.sleep(1)
                
                if cap.isOpened():
                    print(f"✅ Successfully opened video stream at {url}")
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Successfully read frame of shape {frame.shape}")
                        print(f"\n🎉 SUCCESS! Use this URL in your application: {url}")
                        cap.release()
                        return url
                    else:
                        print("❌ Failed to read frame")
                
                cap.release()
            else:
                print(f"❌ Failed to connect to {ip}:8080")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n❌ Could not find a working URL for the camera bridge")
    return None

if __name__ == "__main__":
    print("Testing Windows Camera Bridge")
    
    # First, try the default URL
    success = test_direct_bridge_access()
    
    # If that fails, try alternative URLs
    if not success:
        print("\n⚠️ Default URL (localhost:8080) failed")
        print("⚠️ Trying alternative URLs...")
        working_url = try_alternative_url()
        
        if working_url:
            print(f"\n📌 Found working camera bridge URL: {working_url}")
            print(f"📌 Use this URL in your application!")
        else:
            print("\n❌ Could not connect to the Windows camera bridge")
            print("Make sure the windows_camera_bridge.py script is running on Windows")
            print("You might need to check Windows Firewall settings or try a different URL") 