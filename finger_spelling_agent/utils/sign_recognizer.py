"""
Sign recognition utility for the ASL Finger Spelling Agent.
"""

import os
import logging
import numpy as np
import pickle
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Make TensorFlow optional
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. CNN model type will not be usable.")

logger = logging.getLogger(__name__)

class SignRecognizer:
    """
    ASL fingerspelling recognition using machine learning models.
    """
    
    def __init__(self, model_type="landmarks"):
        """
        Initialize the sign recognizer.
        
        Args:
            model_type (str): Type of model to use - "landmarks" for MediaPipe landmarks
                              or "cnn" for image-based CNN
        """
        # If CNN model requested but TensorFlow not available, fall back to landmarks
        if model_type == "cnn" and not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Falling back to landmarks model type.")
            model_type = "landmarks"
            
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Paths for models
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.landmark_model_path = os.path.join(self.models_dir, "asl_landmark_model.pkl")
        self.landmark_scaler_path = os.path.join(self.models_dir, "asl_landmark_scaler.pkl")
        self.cnn_model_path = os.path.join(self.models_dir, "asl_cnn_model.h5")
        
        # Load the appropriate model
        self._load_model()
        
        logger.info(f"SignRecognizer initialized with model_type={model_type}")
    
    def _load_model(self):
        """
        Load the ML model based on model_type.
        """
        try:
            if self.model_type == "landmarks":
                if os.path.exists(self.landmark_model_path) and os.path.exists(self.landmark_scaler_path):
                    # Load trained model and scaler
                    self.model = joblib.load(self.landmark_model_path)
                    self.scaler = joblib.load(self.landmark_scaler_path)
                    logger.info("Loaded existing landmark-based model")
                else:
                    # Initialize a new model
                    logger.info("No existing landmark model found, initializing a new KNN model")
                    self.model = KNeighborsClassifier(n_neighbors=5)
                    self.scaler = StandardScaler()
            
            elif self.model_type == "cnn" and TENSORFLOW_AVAILABLE:
                if os.path.exists(self.cnn_model_path):
                    # Load trained CNN model
                    self.model = tf.keras.models.load_model(self.cnn_model_path)
                    logger.info("Loaded existing CNN model")
                else:
                    logger.warning("No CNN model found. Please train or download a model.")
            
            else:
                logger.error(f"Invalid model_type: {self.model_type}")
                raise ValueError(f"Invalid model_type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_from_landmarks(self, landmarks):
        """
        Predict the sign from hand landmarks.
        
        Args:
            landmarks (numpy.ndarray): Normalized hand landmarks
            
        Returns:
            dict: Prediction results with letter and confidence
        """
        if self.model_type != "landmarks":
            logger.error("This method can only be used with the 'landmarks' model_type")
            return {"letter": None, "confidence": 0.0}
        
        if landmarks is None or self.model is None:
            return {"letter": None, "confidence": 0.0}
        
        try:
            # Reshape and scale landmarks
            landmarks_reshaped = landmarks.reshape(1, -1)
            
            if self.scaler is not None:
                landmarks_scaled = self.scaler.transform(landmarks_reshaped)
            else:
                landmarks_scaled = landmarks_reshaped
            
            # Get prediction
            prediction = self.model.predict(landmarks_scaled)[0]
            
            # Get prediction probabilities if the model supports it
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(landmarks_scaled)[0]
                confidence = max(proba)
            else:
                # Simple distance-based confidence for KNN
                if hasattr(self.model, "kneighbors"):
                    distances, _ = self.model.kneighbors(landmarks_scaled)
                    # Convert distance to confidence (closer = more confident)
                    avg_distance = np.mean(distances[0])
                    confidence = max(0, 1 - min(avg_distance, 1))
                else:
                    confidence = 0.5  # Default confidence
            
            return {
                "letter": prediction,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error during landmark prediction: {e}")
            return {"letter": None, "confidence": 0.0}
    
    def predict_from_image(self, image):
        """
        Predict the sign from an image.
        
        Args:
            image (numpy.ndarray): Image containing a hand sign
            
        Returns:
            dict: Prediction results with letter and confidence
        """
        if self.model_type != "cnn" or not TENSORFLOW_AVAILABLE:
            logger.error("This method can only be used with the 'cnn' model_type and TensorFlow available")
            return {"letter": None, "confidence": 0.0}
        
        if image is None or self.model is None:
            return {"letter": None, "confidence": 0.0}
        
        try:
            # Preprocess the image
            processed_image = self._preprocess_image(image)
            
            # Get prediction
            predictions = self.model.predict(processed_image)[0]
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            
            return {
                "letter": self.classes[predicted_idx],
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error during image prediction: {e}")
            return {"letter": None, "confidence": 0.0}
    
    def _preprocess_image(self, image):
        """
        Preprocess an image for CNN prediction.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Resize to expected input size
        target_size = (128, 128)
        if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
            import cv2
            image = cv2.resize(image, target_size)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image, axis=0)
    
    def train_landmark_model(self, X, y):
        """
        Train the landmark-based model.
        
        Args:
            X (numpy.ndarray): Features (normalized landmarks)
            y (numpy.ndarray or list): Target labels (letters)
            
        Returns:
            float: Model accuracy
        """
        if self.model_type != "landmarks":
            logger.error("This method can only be used with the 'landmarks' model_type")
            return 0.0
        
        try:
            # Fit the scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train the model
            self.model.fit(X_scaled, y)
            
            # Save the model and scaler
            os.makedirs(self.models_dir, exist_ok=True)
            joblib.dump(self.model, self.landmark_model_path)
            joblib.dump(self.scaler, self.landmark_scaler_path)
            
            # Calculate accuracy
            accuracy = self.model.score(X_scaled, y)
            logger.info(f"Trained landmark model with accuracy: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training landmark model: {e}")
            return 0.0
    
    def create_dummy_landmark_model(self):
        """
        Create a dummy landmark model for testing purposes.
        """
        if self.model_type != "landmarks":
            logger.error("This method can only be used with the 'landmarks' model_type")
            return
        
        try:
            # Create a simple KNN classifier
            self.model = KNeighborsClassifier(n_neighbors=3, weights='distance')
            self.scaler = StandardScaler()
            
            # Generate dummy data for each letter with more distinctive variations
            X = []
            y = []
            
            # For real ASL fingerspelling, each letter has a unique pattern of finger positions
            # We'll simulate this by creating more distinctive patterns for each letter
            for i, letter in enumerate(self.classes):
                # Create multiple samples per letter with variations
                for variation in range(5):  # 5 variations per letter
                    landmarks = np.zeros(63)
                    
                    # Simulate finger positions for different letters
                    # These are very simplified approximations of ASL finger positions
                    
                    # First 21 values (x coordinates of 21 landmarks)
                    for j in range(21):
                        # Different spread patterns for different letters
                        landmarks[j] = 0.1 * np.sin(i * 0.5 + j * 0.3) + 0.05 * variation
                    
                    # Next 21 values (y coordinates)
                    for j in range(21):
                        # Different height patterns for different letters
                        landmarks[21+j] = 0.1 * np.cos(i * 0.3 + j * 0.2) - 0.03 * variation
                    
                    # Last 21 values (z coordinates)
                    for j in range(21):
                        # Different depth patterns
                        landmarks[42+j] = 0.05 * np.sin(i * 0.4 + j * 0.25) + 0.02 * variation
                    
                    # Apply letter-specific patterns
                    # These are crude approximations of some ASL fingerspelling patterns
                    if letter == 'A':  # Fist with thumb to the side
                        landmarks[9:21] = 0.1  # Curl fingers
                        landmarks[3:9] = 0.3   # Extend thumb
                    elif letter == 'B':  # Flat hand, fingers together
                        landmarks[9:21] = 0.5  # Extend fingers
                    elif letter == 'C':  # Curved hand
                        landmarks[9:21] = 0.4  # Curl slightly
                        landmarks[21:42] = landmarks[21:42] + 0.2  # Curve
                    elif letter == 'L':  # Index finger and thumb extended to form an L shape
                        # Reset to default
                        landmarks[9:21] = 0.1  # Curl fingers (middle, ring, pinky)
                        # Extend index finger upward (landmarks 5-8)
                        landmarks[5:9] = 0.6   # Extend index finger
                        landmarks[5+21:9+21] = -0.5  # Index finger points up (y coordinate)
                        # Extend thumb sideways (landmarks 1-4)
                        landmarks[1:5] = 0.7   # Extend thumb
                        landmarks[1+21:5+21] = 0.1  # Thumb points sideways (y coordinate)
                        # Make L shape more pronounced
                        landmarks[1:5] += 0.2 * variation  # Add variation to thumb
                        
                        # Critical: Set middle finger landmarks to be clearly curled (different from V)
                        landmarks[9:13] = 0.05  # Middle finger curled tight
                        landmarks[9+21:13+21] = 0.3  # Middle finger Y position clearly down
                    elif letter == 'V':  # Index and middle fingers in V shape
                        # Reset to default
                        landmarks[13:21] = 0.1  # Curl ring and pinky fingers
                        # Extend index finger upward (landmarks 5-8)
                        landmarks[5:9] = 0.6   # Extend index finger
                        landmarks[5+21:9+21] = -0.6  # Index finger points up (y coordinate)
                        # Extend middle finger upward but at an angle (landmarks 9-12)
                        landmarks[9:13] = 0.6   # Extend middle finger
                        landmarks[9+21:13+21] = -0.55  # Middle finger points up (y coordinate)
                        landmarks[9:13] += 0.15  # Angle the middle finger outward
                        # Curl thumb against palm (different from L)
                        landmarks[1:5] = 0.2    # Thumb position
                    elif letter == 'E':  # Fingers curled, thumb against palm
                        # All fingers curled
                        landmarks[5:21] = 0.15  # Curl all fingers
                        # Thumb position is different from A
                        landmarks[1:5] = 0.2    # Thumb position
                        landmarks[1+21:5+21] = 0.3  # Thumb against palm (y coordinate)
                        # Make E shape more distinctive
                        landmarks[1:5] -= 0.1 * variation  # Add variation to thumb
                    # Other letters would have distinctive patterns too
                    
                    X.append(landmarks)
                    y.append(letter)
            
            # Add extra samples for commonly confused letters to help model distinguish
            # Add more distinctive samples for L vs E
            for _ in range(10):  # Add 10 more samples for L
                landmarks = np.zeros(63)
                
                # Very pronounced L shape - distinctive from E
                # Straight index finger
                landmarks[5:9] = 0.8  # Very extended index
                landmarks[5+21:9+21] = -0.7  # Very upward
                
                # Straight thumb pointing right
                landmarks[1:5] = 0.9  # Very extended thumb
                landmarks[1+21:5+21] = 0.05  # Horizontal
                
                # Other fingers curled
                landmarks[9:21] = 0.05
                
                # Add slight variation
                landmarks += np.random.normal(0, 0.05, 63)
                
                X.append(landmarks)
                y.append('L')
            
            for _ in range(10):  # Add 10 more samples for E
                landmarks = np.zeros(63)
                
                # Very pronounced E shape - distinctive from L
                # All fingers curled tightly
                landmarks[5:21] = 0.1
                
                # Thumb pressed against fingers (not extended like L)
                landmarks[1:5] = 0.15
                landmarks[1+21:5+21] = 0.35
                
                # Add slight variation
                landmarks += np.random.normal(0, 0.05, 63)
                
                X.append(landmarks)
                y.append('E')
            
            # Add more distinctive samples for L vs V
            for _ in range(15):  # Add 15 more samples for L specifically to distinguish from V
                landmarks = np.zeros(63)
                
                # Very pronounced L shape - distinctive from V
                # Straight index finger
                landmarks[5:9] = 0.8  # Very extended index
                landmarks[5+21:9+21] = -0.7  # Very upward
                
                # Straight thumb pointing right
                landmarks[1:5] = 0.9  # Very extended thumb
                landmarks[1+21:5+21] = 0.05  # Horizontal
                
                # Middle finger clearly curled (this is the main differentiator from V)
                landmarks[9:13] = 0.03  # Middle finger tightly curled
                landmarks[9+21:13+21] = 0.4  # Middle finger clearly down
                
                # Other fingers curled
                landmarks[13:21] = 0.05
                
                # Add slight variation
                landmarks += np.random.normal(0, 0.03, 63)
                
                X.append(landmarks)
                y.append('L')
            
            for _ in range(15):  # Add 15 more samples for V
                landmarks = np.zeros(63)
                
                # Very pronounced V shape - distinctive from L
                # Straight index finger
                landmarks[5:9] = 0.8  # Very extended index
                landmarks[5+21:9+21] = -0.7  # Very upward
                
                # Straight middle finger also extended up but at angle
                landmarks[9:13] = 0.78  # Middle finger extended
                landmarks[9+21:13+21] = -0.68  # Middle finger up
                landmarks[9:13] += 0.2  # Angle outward for V shape
                
                # Thumb curled (main differentiator from L)
                landmarks[1:5] = 0.2  # Thumb not extended
                landmarks[1+21:5+21] = 0.25  # Thumb not to side
                
                # Other fingers curled
                landmarks[13:21] = 0.05
                
                # Add slight variation
                landmarks += np.random.normal(0, 0.03, 63)
                
                X.append(landmarks)
                y.append('V')
            
            # Train on dummy data
            X = np.array(X)
            self.train_landmark_model(X, y)
            
            logger.info("Created dummy landmark model for testing with improved letter distinctions")
            
        except Exception as e:
            logger.error(f"Error creating dummy model: {e}")
            logger.exception(e)


def test_sign_recognizer():
    """
    Simple test function for the SignRecognizer class.
    """
    # Create a sign recognizer with landmark-based model
    recognizer = SignRecognizer(model_type="landmarks")
    
    # Create a dummy model for testing
    recognizer.create_dummy_landmark_model()
    
    # Test prediction with some dummy landmarks
    dummy_landmarks = np.zeros(63)
    dummy_landmarks[0] = 0  # Should predict 'A'
    
    result = recognizer.predict_from_landmarks(dummy_landmarks)
    print(f"Prediction: {result['letter']}, Confidence: {result['confidence']:.2f}")
    
    # Try another one
    dummy_landmarks[0] = 25/26.0  # Should predict 'Z'
    result = recognizer.predict_from_landmarks(dummy_landmarks)
    print(f"Prediction: {result['letter']}, Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    test_sign_recognizer() 