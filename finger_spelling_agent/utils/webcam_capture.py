"""
Webcam capture utility for the ASL Finger Spelling Agent.
"""

import cv2
import time
import logging
import numpy as np
import threading
import os
import platform

# Import our custom webcam utilities for WSL compatibility
from .webcam_utils import get_camera_source, is_wsl, get_webcam_instructions

logger = logging.getLogger(__name__)

class WebcamCapture:
    """
    Class for capturing and managing webcam feed.
    """
    
    def __init__(self, camera_id=None, width=640, height=480, fps=30):
        """
        Initialize the webcam capture.
        
        Args:
            camera_id (int or str, optional): Camera device ID or URL
                If None, it will be auto-detected based on environment
            width (int): Desired frame width
            height (int): Desired frame height
            fps (int): Desired frames per second
        """
        # Auto-detect camera source if not specified
        self.camera_id = camera_id if camera_id is not None else get_camera_source()
        
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Check if running in WSL
        self.in_wsl = is_wsl()
        if self.in_wsl:
            logger.info("Running in WSL environment. Camera access may require additional setup.")
            instructions = get_webcam_instructions()
            if instructions:
                logger.info(instructions)
    
    def set_camera_source(self, camera_source):
        """
        Manually set the camera source.
        
        Args:
            camera_source (int or str): Camera ID or URL to use
            
        Returns:
            bool: True if the source was set, False if already running
        """
        if self.running:
            logger.warning("Cannot change camera source while running")
            return False
            
        self.camera_id = camera_source
        logger.info(f"Camera source manually set to: {camera_source}")
        return True
    
    def start(self):
        """
        Start capturing frames from the webcam in a separate thread.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Webcam capture is already running")
            return True
        
        # Initialize capture device
        logger.info(f"Opening camera (source: {self.camera_id})")
        
        # Handle different camera source types
        if isinstance(self.camera_id, str) and self.camera_id.startswith('http'):
            # HTTP stream (likely from our Windows bridge)
            self.cap = cv2.VideoCapture(self.camera_id)
            logger.info(f"Using HTTP stream: {self.camera_id}")
        else:
            # Regular camera device
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Try to set camera properties (may not work with HTTP streams)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            error_msg = f"Failed to open camera (ID: {self.camera_id})"
            
            if self.in_wsl:
                error_msg += "\n\nRunning in WSL: Camera access requires additional setup."
                error_msg += "\nSee the instructions in the logs or check webcam_utils.py."
                
            logger.error(error_msg)
            return False
        
        # Get actual camera properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera opened successfully")
        logger.info(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
        
        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Webcam capture started")
        return True
    
    def stop(self):
        """Stop capturing frames and release resources."""
        if not self.running:
            return
        
        logger.info("Stopping webcam capture")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.frame = None
        
        logger.info("Webcam capture stopped")
    
    def get_frame(self):
        """
        Get the current frame from the webcam.
        
        Returns:
            numpy.ndarray or None: Current frame or None if not available
        """
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def _capture_loop(self):
        """Continuously capture frames from the webcam."""
        frame_count = 0
        start_time = time.time()
        fps_report_interval = 5.0  # Report FPS every 5 seconds
        
        while self.running and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Update the current frame (thread-safe)
            with self.lock:
                self.frame = frame
            
            # FPS calculation and logging
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= fps_report_interval:
                fps = frame_count / elapsed
                logger.debug(f"WebcamCapture: {fps:.2f} FPS")
                frame_count = 0
                start_time = time.time()
            
            # Sleep to control CPU usage
            time.sleep(1.0 / (self.fps * 2))

# Add a helper function to test webcam access
def test_webcam(camera_source=None, duration=5):
    """
    Test webcam access by trying to capture and display frames.
    
    Args:
        camera_source (int or str, optional): Camera source to use
        duration (int): Test duration in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Testing webcam access: {camera_source if camera_source is not None else 'auto-detect'}")
    
    # Create webcam capture instance
    webcam = WebcamCapture(camera_id=camera_source)
    
    # Try to start the webcam
    if not webcam.start():
        print("Failed to start webcam")
        return False
    
    print("Webcam started successfully")
    print(f"Running test for {duration} seconds...")
    
    # Create a named window that can fit the output
    cv2.namedWindow("Webcam Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Test", 640, 480)
    
    start_time = time.time()
    frames_read = 0
    
    try:
        while time.time() - start_time < duration:
            # Get a frame
            frame = webcam.get_frame()
            
            if frame is not None:
                frames_read += 1
                
                # Draw some info on the frame
                cv2.putText(frame, f"Frames: {frames_read}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elapsed = time.time() - start_time
                cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                fps = frames_read / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow("Webcam Test", frame)
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame received")
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        # Clean up
        webcam.stop()
        cv2.destroyAllWindows()
    
    print(f"Test completed: {frames_read} frames captured in {time.time() - start_time:.1f} seconds")
    return frames_read > 0

if __name__ == "__main__":
    test_webcam() 