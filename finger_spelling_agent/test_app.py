"""
ASL Finger Spelling Agent - Test Application with Camera URL Support

This version of the app handles the camera_url through environment variables
and initializes the WebcamCapture with the specified URL.
"""

import os
import time
import logging
import cv2
import numpy as np
import threading
import streamlit as st
import argparse
import sys
from dotenv import load_dotenv

# Import utility modules
from utils.webcam_capture import WebcamCapture
from utils.hand_detector import HandDetector
from utils.sign_recognizer import SignRecognizer
from utils.visualization import draw_landmarks, draw_prediction_results

# Import the original app
from app import FingerSpellingApp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TestFingerSpellingApp(FingerSpellingApp):
    """
    Extended version of FingerSpellingApp that supports camera URL
    """
    
    def __init__(self, camera_url=None):
        """
        Initialize the application components.
        
        Args:
            camera_url (str, optional): URL for the camera feed
        """
        # Call parent initializer
        super().__init__()
        
        # Store camera URL
        self.camera_url = camera_url
        logger.info(f"TestFingerSpellingApp initialized with camera_url: {camera_url}")
    
    def initialize_components(self):
        """
        Initialize webcam, detector, and recognizer components.
        """
        try:
            # Initialize webcam with camera URL if provided
            if self.camera_url:
                self.webcam = WebcamCapture(camera_id=self.camera_url, width=640, height=480, fps=30)
                logger.info(f"Using camera URL: {self.camera_url}")
            else:
                self.webcam = WebcamCapture(width=640, height=480, fps=30)
                logger.info("Using default camera source")
            
            # Initialize hand detector
            self.hand_detector = HandDetector(
                static_mode=False,
                max_hands=1,
                min_detection_confidence=self.detection_confidence
            )
            
            # Initialize sign recognizer
            self.sign_recognizer = SignRecognizer(model_type=self.model_type)
            
            # Create a dummy model for testing if needed
            if self.model_type == "landmarks":
                self.sign_recognizer.create_dummy_landmark_model()
            
            logger.info("Components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False


# Main Streamlit app code
def main():
    # Get camera URL from environment variable
    camera_url = os.environ.get('CAMERA_URL')
    
    # Debug output
    print("DEBUG - Using CAMERA_URL environment variable:", camera_url)
    
    # Set page config
    st.set_page_config(
        page_title="ASL Finger Spelling Agent",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    if 'app_running' not in st.session_state:
        st.session_state.app_running = False
        
    if 'target_letter' not in st.session_state:
        st.session_state.target_letter = None
        
    if 'practice_mode' not in st.session_state:
        st.session_state.practice_mode = "guided"
        
    if 'correct_count' not in st.session_state:
        st.session_state.correct_count = 0
        
    if 'total_attempts' not in st.session_state:
        st.session_state.total_attempts = 0
    
    # Create app instance
    if 'app' not in st.session_state:
        st.session_state.app = TestFingerSpellingApp(camera_url=camera_url)
    
    app = st.session_state.app
    
    # App title and description
    st.title("👋 ASL Finger Spelling Agent")
    st.subheader("Practice American Sign Language fingerspelling with real-time feedback")
    
    # Display camera info
    if camera_url:
        st.success(f"Using camera URL: {camera_url}")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Practice mode
        practice_mode = st.radio(
            "Practice Mode",
            ["Guided Practice", "Free Practice"],
            index=0 if st.session_state.practice_mode == "guided" else 1
        )
        st.session_state.practice_mode = "guided" if practice_mode == "Guided Practice" else "free"
        
        # Detection settings
        st.subheader("Detection Settings")
        
        detection_confidence = st.slider(
            "Hand Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=app.detection_confidence,
            step=0.05
        )
        
        if detection_confidence != app.detection_confidence:
            app.set_detection_confidence(detection_confidence)
        
        # Model selection
        model_type = st.selectbox(
            "Recognition Model",
            ["MediaPipe Landmarks", "CNN Classifier"],
            index=0 if app.model_type == "landmarks" else 1
        )
        
        selected_model = "landmarks" if model_type == "MediaPipe Landmarks" else "cnn"
        if selected_model != app.model_type:
            app.set_model_type(selected_model)
        
        # Start/stop button
        if st.button("Start" if not st.session_state.app_running else "Stop"):
            if not st.session_state.app_running:
                if app.start():
                    st.session_state.app_running = True
            else:
                app.stop()
                st.session_state.app_running = False
        
        # Status indicator
        st.markdown("---")
        st.subheader("Status")
        if st.session_state.app_running:
            st.success("Application is running")
        else:
            st.warning("Application is stopped")
        
        # Instructions
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("""
        1. Select your practice mode
        2. Click 'Start' to begin
        3. Position your hand in the camera view
        4. Make ASL fingerspelling signs
        5. Get real-time feedback
        """)
        
        # Progress display
        if st.session_state.practice_mode == "guided":
            st.markdown("---")
            st.subheader("Progress")
            if st.session_state.total_attempts > 0:
                accuracy = (st.session_state.correct_count / st.session_state.total_attempts) * 100
                st.write(f"Accuracy: {accuracy:.1f}%")
                st.write(f"Correct: {st.session_state.correct_count} / {st.session_state.total_attempts}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Webcam feed
        st.header("Webcam View")
        frame_placeholder = st.empty()
        
        # Info text if not running
        if 'app_running' not in st.session_state:
            st.session_state.app_running = False
            
        if not st.session_state.app_running:
            frame_placeholder.info("Click 'Start' in the sidebar to begin.")
    
    with col2:
        if st.session_state.practice_mode == "guided":
            # Guided practice
            st.header("Guided Practice")
            
            # Target letter
            if st.session_state.target_letter is None or st.button("Next Letter"):
                # Get random letter
                import random
                letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                st.session_state.target_letter = letter
                app.set_target_letter(letter)
            
            # Display current target
            st.subheader(f"Sign the letter: {st.session_state.target_letter}")
            
            # Check button
            if st.button("Check"):
                prediction = app.get_current_prediction()
                if prediction and prediction.get('letter'):
                    predicted_letter = prediction.get('letter')
                    correct = predicted_letter == st.session_state.target_letter
                    
                    if correct:
                        st.success(f"Correct! You signed '{predicted_letter}' correctly.")
                        st.session_state.correct_count += 1
                    else:
                        st.error(f"Not quite. You signed '{predicted_letter}' but should be '{st.session_state.target_letter}'.")
                    
                    st.session_state.total_attempts += 1
                else:
                    st.warning("No hand sign detected. Make sure your hand is visible.")
        
        else:
            # Free practice
            st.header("Free Practice")
            st.markdown("Practice any letter you want. The system will recognize your signs in real-time.")
            
            # Current prediction
            pred_placeholder = st.empty()
    
    # Main update loop for Streamlit
    if st.session_state.app_running:
        # Use a placeholder to update the frame
        while True:
            try:
                # Get the current frame
                frame = app.get_current_frame()
                
                if frame is not None:
                    # Display the frame
                    frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                # Update prediction display in free practice mode
                if st.session_state.practice_mode == "free":
                    prediction = app.get_current_prediction()
                    if prediction and prediction.get('letter'):
                        letter = prediction.get('letter')
                        confidence = prediction.get('confidence', 0.0)
                        
                        if confidence > 0.7:
                            pred_placeholder.success(f"Detected: {letter} ({confidence:.2f})")
                        elif confidence > 0.5:
                            pred_placeholder.info(f"Detected: {letter} ({confidence:.2f})")
                        else:
                            pred_placeholder.warning(f"Detected: {letter} ({confidence:.2f})")
                    else:
                        pred_placeholder.warning("No hand sign detected")
                
                # Short pause to allow Streamlit to update
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in Streamlit update loop: {e}")
                st.error(f"Error: {e}")
                break
            
            # Check if the app is still running
            if 'app_running' in st.session_state and not st.session_state.app_running:
                break


if __name__ == "__main__":
    main() 