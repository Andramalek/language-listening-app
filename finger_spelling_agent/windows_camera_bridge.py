"""
Windows Camera Bridge for WSL

This script should be run from Windows (not WSL) to capture webcam frames
and serve them to WSL applications via HTTP.

Prerequisites:
- Install Python on Windows
- Install opencv-python and flask packages:
  pip install opencv-python flask

Usage:
1. Run this script on Windows: python windows_camera_bridge.py
2. Access the camera in WSL through the URL displayed in the console
   (e.g., http://YOUR_IP:8080/video_feed)
"""

import cv2
import time
import threading
import socket
import os
import sys
from flask import Flask, Response, request

app = Flask(__name__)

# Global variables
camera = None
frame = None
stop_event = threading.Event()
camera_id = 0  # Default camera ID
port = 8080    # Using 8080 instead of 5000 (which is often blocked)

def get_all_ip_addresses():
    """Get all IP addresses of this machine."""
    ip_list = []
    
    try:
        # Get hostname
        hostname = socket.gethostname()
        ip_list.append(("Hostname", hostname))
        
        # Get local IP by hostname
        try:
            local_ip = socket.gethostbyname(hostname)
            ip_list.append(("Local IP (by hostname)", local_ip))
        except:
            pass
            
        # Get all network interfaces
        try:
            addresses = socket.getaddrinfo(hostname, None)
            for addr in addresses:
                ip = addr[4][0]
                if ip not in [i[1] for i in ip_list] and not ip.startswith('127.'):
                    ip_list.append(("Network interface", ip))
        except:
            pass
            
        # Special handling for IPv4
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # This doesn't make a real connection
            s.connect(('8.8.8.8', 1))
            ip = s.getsockname()[0]
            if ip not in [i[1] for i in ip_list]:
                ip_list.append(("Primary IP (outbound)", ip))
        except:
            pass
        finally:
            s.close()
            
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
        
    return ip_list

def test_cameras():
    """Test available cameras and return the first working one."""
    print("Scanning for available cameras...")
    
    for i in range(5):  # Try first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✓ Camera {i} works: {width}x{height} @ {fps} FPS")
            cap.release()
            return i
        else:
            print(f"✗ Camera {i} not available")
            cap.release()
    
    return 0  # Default to 0 if no cameras found

def capture_frames():
    """Continuously capture frames from the webcam."""
    global camera, frame, stop_event, camera_id
    
    # First try the specified camera_id
    camera = cv2.VideoCapture(camera_id)
    
    # If that fails, test other cameras
    if not camera.isOpened():
        print(f"Could not open camera {camera_id}, scanning for available cameras...")
        camera_id = test_cameras()
        camera = cv2.VideoCapture(camera_id)
    
    # Try to set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    if not camera.isOpened():
        print("ERROR: Could not open any camera.")
        print("Please make sure a webcam is connected and not in use by another application.")
        return
    
    print("\n✓ Camera opened successfully!")
    print(f"   Camera ID: {camera_id}")
    print(f"   Resolution: {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"   FPS: {camera.get(cv2.CAP_PROP_FPS)}")
    
    # Read frames in a loop
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        success, current_frame = camera.read()
        
        if not success:
            print("Error: Failed to read frame from camera.")
            time.sleep(0.1)
            continue
            
        # Update the global frame
        frame = current_frame
        
        # Report FPS occasionally
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 5.0:  # Report every 5 seconds
            fps = frame_count / elapsed
            print(f"Camera capturing at {fps:.1f} FPS")
            frame_count = 0
            start_time = time.time()
            
        # Control the capture rate
        time.sleep(1/30)  # Limit to around 30 FPS

def generate_frames():
    """Generate frames for the HTTP response."""
    global frame
    
    # Add a text overlay with connection info
    def add_overlay(img):
        if img is None:
            return None
        
        # Make a copy to avoid modifying the original
        img_copy = img.copy()
        
        # Add text with IP and port
        text = f"WSL Camera Bridge - Connected"
        cv2.putText(img_copy, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                   
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img_copy, timestamp, (10, img_copy.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return img_copy
    
    # Frame generation loop
    while not stop_event.is_set():
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Add overlay to the frame
        output_frame = add_overlay(frame)
            
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame)
        if not ret:
            continue
            
        # Convert to bytes and yield
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1/30)  # Limit to around 30 FPS

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    # Log connection info
    client_ip = request.remote_addr
    print(f"Camera feed accessed from: {client_ip}")
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ping')
def ping():
    """Simple endpoint to test connectivity."""
    return "Windows Camera Bridge is running!"

@app.route('/')
def index():
    """Home page with instructions."""
    
    # Get the requested host
    host = request.host
    
    return f"""
    <html>
        <head>
            <title>Windows Camera Bridge for WSL</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                h1 {{ color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .success {{ color: green; font-weight: bold; }}
                .camera-view {{ border: 1px solid #ddd; border-radius: 5px; margin-top: 20px; }}
                .url-box {{ background-color: #e9f7fe; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .important {{ color: #d9534f; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Windows Camera Bridge for WSL</h1>
                <p class="success">✓ Bridge is running successfully!</p>
                <p>Your camera is now accessible through HTTP from WSL and other devices on your network.</p>
                
                <div class="url-box">
                    <h3>Camera Feed URL:</h3>
                    <pre>http://{host}/video_feed</pre>
                </div>
                
                <h2>Usage in WSL:</h2>
                <p>To access the camera stream, use one of the following methods:</p>
                
                <h3>1. Test the connection:</h3>
                <pre>curl http://{host}/ping</pre>
                
                <h3>2. In Python with OpenCV:</h3>
                <pre>
import cv2
cap = cv2.VideoCapture('http://{host}/video_feed')
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
                </pre>
                
                <p class="important">Note: Keep this window open while using the camera in WSL.</p>
                
                <h3>3. Live Preview:</h3>
                <div class="camera-view">
                    <img src="/video_feed" width="640" height="480" />
                </div>
            </div>
        </body>
    </html>
    """

def print_connection_instructions(port):
    """Print connection instructions with all possible IP addresses."""
    print("\n" + "="*50)
    print(" WINDOWS CAMERA BRIDGE - CONNECTION INFORMATION")
    print("="*50)
    
    ip_addresses = get_all_ip_addresses()
    
    print("\nYou can access the camera using ANY of these URLs:")
    print("(Try them in order until one works in WSL)")
    
    # Always show localhost first
    print(f"• http://localhost:{port}/video_feed")
    
    # Then show all detected IPs
    for ip_type, ip in ip_addresses:
        print(f"• http://{ip}:{port}/video_feed  ({ip_type})")
    
    # Also show hostname
    hostname = socket.gethostname()
    print(f"• http://{hostname}:{port}/video_feed  (Hostname)")
    
    print("\nIn WSL, run this test command:")
    print("python manual_camera_test.py --url \"http://YOUR-IP-FROM-LIST:8080/video_feed\"")
    
    print("\n" + "="*50)
    print(" TROUBLESHOOTING")
    print("="*50)
    print("• If connections fail, check your Windows Firewall settings")
    print("• Make sure port 8080 is allowed through the firewall")
    print("• Try different IP addresses from the list above")
    print("• Keep this window open while using the camera")
    print("="*50 + "\n")

if __name__ == '__main__':
    # Check command line arguments for camera ID
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        camera_id = int(sys.argv[1])
        print(f"Using specified camera ID: {camera_id}")
    
    # Start frame capture in a separate thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Give the camera time to initialize
    time.sleep(2)
    
    try:
        print_connection_instructions(port)
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=port, threaded=True)
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        stop_event.set()
        if camera is not None and camera.isOpened():
            camera.release()
        print("Camera released. Server stopped.")
        
    # Add a pause at the end so the window doesn't close immediately if run by double-clicking
    input("\nPress Enter to exit...") 