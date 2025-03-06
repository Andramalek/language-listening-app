"""
Hand detection utility using MediaPipe for the ASL Finger Spelling Agent.
"""

import cv2
import logging
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

class HandDetector:
    """
    Hand detection and landmark tracking using MediaPipe Hands.
    """
    
    def __init__(self, static_mode=False, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize the hand detector.
        
        Args:
            static_mode (bool): If True, detection runs on every frame. If False, detection runs once and then tracking is used
            max_hands (int): Maximum number of hands to detect
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for landmark tracking
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"HandDetector initialized (max_hands={max_hands}, detection_confidence={min_detection_confidence})")
    
    def find_hands(self, image, draw=True):
        """
        Detect hands and landmarks in an image.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            draw (bool): Whether to draw landmarks on the image
            
        Returns:
            numpy.ndarray: Image with landmarks drawn if draw=True
            list: List of detected hands with landmarks
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        
        hands_data = []
        
        # Process detection results
        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # Get hand information
                hand_info = {}
                
                # Determine if it's left or right hand
                if self.results.multi_handedness:
                    handedness = self.results.multi_handedness[hand_idx]
                    hand_info['type'] = handedness.classification[0].label
                    hand_info['confidence'] = handedness.classification[0].score
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z, 'pixel_x': cx, 'pixel_y': cy})
                
                hand_info['landmarks'] = landmarks
                hands_data.append(hand_info)
                
                # Draw landmarks on the image
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        return image, hands_data
    
    def extract_landmarks(self, hands_data):
        """
        Extract normalized landmarks from hands data in a format suitable for ML models.
        
        Args:
            hands_data (list): List of detected hands with landmarks
            
        Returns:
            numpy.ndarray or None: Normalized landmarks for the first detected hand, None if no hand detected
        """
        if not hands_data:
            return None
        
        # Get the first hand (assuming max_hands=1 for ASL fingerspelling)
        hand = hands_data[0]
        landmarks = hand['landmarks']
        
        # Extract normalized (x, y, z) coordinates
        landmark_array = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Normalize relative to wrist position
        wrist_pos = landmark_array[0]
        normalized_landmarks = landmark_array - wrist_pos
        
        # Flatten the array for ML input
        return normalized_landmarks.flatten()
    
    def close(self):
        """
        Release resources.
        """
        self.hands.close()
        logger.info("HandDetector resources released")


def test_hand_detector():
    """
    Simple test function for the HandDetector class.
    """
    import cv2
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Find hands and landmarks
            frame, hands_data = detector.find_hands(frame)
            
            # Display number of hands detected
            num_hands = len(hands_data)
            cv2.putText(frame, f"Hands: {num_hands}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Extract landmarks if hand is detected
            if hands_data:
                landmarks = detector.extract_landmarks(hands_data)
                if landmarks is not None:
                    # Just print the shape for demonstration
                    print(f"Landmarks shape: {landmarks.shape}")
            
            # Display the frame
            cv2.imshow("Hand Detector Test", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_hand_detector() 