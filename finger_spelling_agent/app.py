"""
ASL Finger Spelling Agent - Integrated Application

This version of the app integrates webcam input, hand detection, and sign recognition
for a real-time ASL fingerspelling practice experience.
"""

import os
import time
import logging
import cv2
import numpy as np
import threading
import streamlit as st
from dotenv import load_dotenv

# Import utility modules
from utils.webcam_capture import WebcamCapture
from utils.hand_detector import HandDetector
from utils.sign_recognizer import SignRecognizer

from utils.visualization import draw_landmarks, draw_prediction_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FingerSpellingApp:
    """
    Integrated application for ASL fingerspelling practice.
    """
    
    def __init__(self):
        """
        Initialize the application components.
        """
        self.webcam = None
        self.hand_detector = None
        self.sign_recognizer = None
        
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.current_frame = None
        self.current_hands_data = None
        self.current_prediction = None
        
        self.target_letter = None
        self.show_landmarks = True
        
        # Configuration
        self.detection_confidence = 0.7
        self.model_type = "landmarks"  # "landmarks" or "cnn"
        
        logger.info("FingerSpellingApp initialized")
    
    def initialize_components(self):
        """
        Initialize webcam, detector, and recognizer components.
        """
        try:
            # Initialize webcam
            self.webcam = WebcamCapture(width=640, height=480, fps=30)
            
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
    
    def start(self):
        """
        Start the application.
        """
        if self.is_running:
            logger.warning("Application is already running")
            return True
        
        # Initialize components if not already done
        if self.webcam is None:
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
        
        # Start webcam
        if not self.webcam.start():
            logger.error("Failed to start webcam")
            return False
        
        # Start processing thread
        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("FingerSpellingApp started")
        return True
    
    def stop(self):
        """
        Stop the application.
        """
        self.is_running = False
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        if self.webcam is not None:
            self.webcam.stop()
        
        if self.hand_detector is not None:
            self.hand_detector.close()
        
        logger.info("FingerSpellingApp stopped")
    
    def _processing_loop(self):
        """
        Main processing loop that runs in a separate thread.
        """
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            try:
                # Get frame from webcam
                frame = self.webcam.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Detect hands and landmarks
                annotated_frame, hands_data = self.hand_detector.find_hands(
                    frame, draw=self.show_landmarks
                )
                
                # Process landmarks for sign recognition
                prediction = None
                if hands_data:
                    landmarks = self.hand_detector.extract_landmarks(hands_data)
                    
                    if landmarks is not None:
                        if self.model_type == "landmarks":
                            prediction = self.sign_recognizer.predict_from_landmarks(landmarks)
                        elif self.model_type == "cnn":
                            # Extract hand region for CNN
                            # This is simplified and would need to be improved for real use
                            h, w, _ = frame.shape
                            hand_img = frame[max(0, int(h/4)):min(h, int(3*h/4)), max(0, int(w/4)):min(w, int(3*w/4))]
                            prediction = self.sign_recognizer.predict_from_image(hand_img)
                
                # Draw prediction results
                if prediction:
                    annotated_frame = draw_prediction_results(
                        annotated_frame, prediction, target_letter=self.target_letter
                    )
                
                # Update current data with thread safety
                with self.lock:
                    self.current_frame = annotated_frame
                    self.current_hands_data = hands_data
                    self.current_prediction = prediction
                
                # Calculate and log FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 5.0:  # Log FPS every 5 seconds
                    fps = frame_count / elapsed_time
                    logger.debug(f"Processing FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()
                
                # Control processing rate
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def get_current_frame(self):
        """
        Get the current processed frame.
        
        Returns:
            numpy.ndarray or None: The current frame if available
        """
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()
    
    def get_current_prediction(self):
        """
        Get the current prediction.
        
        Returns:
            dict or None: The current prediction if available
        """
        with self.lock:
            return self.current_prediction
    
    def set_target_letter(self, letter):
        """
        Set the target letter for practice.
        
        Args:
            letter (str): The target letter
        """
        self.target_letter = letter
        logger.info(f"Target letter set to: {letter}")
    
    def set_detection_confidence(self, confidence):
        """
        Set the hand detection confidence threshold.
        
        Args:
            confidence (float): Confidence threshold (0.0 to 1.0)
        """
        self.detection_confidence = confidence
        if self.hand_detector is not None:
            self.hand_detector.min_detection_confidence = confidence
        logger.info(f"Detection confidence set to: {confidence}")
    
    def set_model_type(self, model_type):
        """
        Set the model type for sign recognition.
        
        Args:
            model_type (str): "landmarks" or "cnn"
        """
        if model_type not in ["landmarks", "cnn"]:
            logger.error(f"Invalid model_type: {model_type}")
            return
        
        self.model_type = model_type
        
        # Reinitialize sign recognizer
        if self.sign_recognizer is not None:
            self.sign_recognizer = SignRecognizer(model_type=model_type)
            
            # Create a dummy model for testing if needed
            if model_type == "landmarks":
                self.sign_recognizer.create_dummy_landmark_model()
        
        logger.info(f"Model type set to: {model_type}")


# Main Streamlit app code
def main():
    # Set page config
    st.set_page_config(
        page_title="ASL Finger Spelling Agent",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create session state variables
    if 'app' not in st.session_state:
        st.session_state.app = FingerSpellingApp()
    
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
    
    app = st.session_state.app
    
    # App title and description
    st.title("👋 ASL Finger Spelling Agent")
    st.subheader("Practice American Sign Language fingerspelling with real-time feedback")
    
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
            if not st.session_state.app_running:
                break


if __name__ == "__main__":
    main() 