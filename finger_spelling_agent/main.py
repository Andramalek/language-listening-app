"""
Finger Spelling Agent - ASL Learning Application

A tool for practicing American Sign Language (ASL) fingerspelling using webcam input.
"""

import os
import time
import logging
import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Import custom modules
from utils.webcam_capture import WebcamCapture
from utils.hand_detector import HandDetector
from utils.sign_recognizer import SignRecognizer
from utils.visualization import draw_landmarks, draw_prediction_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="ASL Finger Spelling Agent",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_letter' not in st.session_state:
    st.session_state.current_letter = None
if 'correct_count' not in st.session_state:
    st.session_state.correct_count = 0
if 'total_attempts' not in st.session_state:
    st.session_state.total_attempts = 0
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'practice_mode' not in st.session_state:
    st.session_state.practice_mode = "guided"  # "guided" or "free"

# Function to get a random letter
def get_random_letter():
    import random
    return random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Function to check prediction
def check_prediction(prediction, target):
    return prediction == target

# Main application
st.title("👋 ASL Finger Spelling Agent")
st.subheader("Practice American Sign Language fingerspelling")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Practice mode selection
    practice_mode = st.radio(
        "Practice Mode",
        ["Guided Practice", "Free Practice"],
        index=0
    )
    st.session_state.practice_mode = "guided" if practice_mode == "Guided Practice" else "free"
    
    # Model selection
    model_option = st.selectbox(
        "Recognition Model",
        ["MediaPipe Landmarks", "CNN Classifier"],
        index=0
    )
    
    # Detection sensitivity
    detection_confidence = st.slider(
        "Hand Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    # Start/stop webcam
    webcam_button = st.button(
        "Start Webcam" if not st.session_state.webcam_active else "Stop Webcam"
    )
    
    if webcam_button:
        st.session_state.webcam_active = not st.session_state.webcam_active
        if st.session_state.webcam_active:
            st.success("Webcam activated")
        else:
            st.warning("Webcam stopped")
    
    # Instructions
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Select a practice mode
    2. Click 'Start Webcam'
    3. Show your hand to the camera
    4. Practice fingerspelling the displayed letter
    5. Receive feedback on your signing
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Webcam view
    st.header("Webcam View")
    webcam_placeholder = st.empty()
    
    # This will be replaced with actual webcam feed
    if not st.session_state.webcam_active:
        webcam_placeholder.info("Click 'Start Webcam' in the sidebar to begin.")
    else:
        # Placeholder for webcam feed - will be implemented with actual webcam integration
        webcam_placeholder.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/American_Sign_Language_ASL.svg/1200px-American_Sign_Language_ASL.svg.png",
            caption="Webcam will appear here",
            width=400
        )

with col2:
    # Practice area
    st.header("Practice Area")
    
    if st.session_state.practice_mode == "guided":
        # Display a letter to practice
        if st.session_state.current_letter is None or st.button("Next Letter"):
            st.session_state.current_letter = get_random_letter()
        
        st.markdown(f"### Sign the letter: **{st.session_state.current_letter}**")
        
        # Display reference image for the current letter
        st.image(
            f"https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Sign_language_{st.session_state.current_letter.lower()}.svg/200px-Sign_language_{st.session_state.current_letter.lower()}.svg.png",
            caption=f"ASL Fingerspelling for '{st.session_state.current_letter}'",
            width=150
        )
        
        # Status and feedback
        st.markdown("### Status")
        st.markdown(f"Correct: {st.session_state.correct_count} / {st.session_state.total_attempts}")
        
        if st.session_state.feedback:
            if "Correct" in st.session_state.feedback:
                st.success(st.session_state.feedback)
            else:
                st.warning(st.session_state.feedback)
    
    else:  # Free practice mode
        st.markdown("### Free Practice Mode")
        st.markdown("Practice any letter you want. The system will recognize and provide feedback.")
        
        # Display all letter references
        st.markdown("### ASL Alphabet Reference")
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/American_Sign_Language_ASL.svg/1200px-American_Sign_Language_ASL.svg.png",
            caption="ASL Fingerspelling Alphabet",
            width=300
        )
        
        # Detected letter
        st.markdown("### Detected Letter")
        detected_letter_placeholder = st.empty()
        detected_letter_placeholder.markdown("Make a sign to detect")

# Note: The actual webcam integration and model inference will be implemented
# in separate utility files that we'll create next
st.markdown("---")
st.markdown("### About ASL Fingerspelling")
st.markdown("""
American Sign Language (ASL) fingerspelling uses a single hand to spell out English words letter by letter.
This app helps you practice the ASL alphabet using computer vision to provide real-time feedback.

**Note:** ASL is a rich, complex language with its own grammar and syntax.
Fingerspelling is just one small part of ASL communication.
""")

if __name__ == "__main__":
    # The main application logic is handled by Streamlit's interactive elements
    # The actual webcam integration and processing will be implemented in the utility modules
    pass 