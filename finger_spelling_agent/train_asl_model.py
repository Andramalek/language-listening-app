"""
Train ASL Model from Dataset

This script trains an ASL fingerspelling model using the processed landmarks
from the downloaded dataset.
"""

import os
import sys
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import our sign recognizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sign_recognizer import SignRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(landmarks_path, labels_path, output_path, test_size=0.2, random_state=42):
    """
    Train a new model using the processed dataset.
    
    Args:
        landmarks_path (str): Path to the landmarks file
        labels_path (str): Path to the labels file
        output_path (str): Path to save the trained model
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        bool: True if training was successful, False otherwise
    """
    try:
        # Load the dataset
        logger.info(f"Loading dataset from {landmarks_path} and {labels_path}")
        X = np.load(landmarks_path)
        y = np.load(labels_path)
        
        # Check if the dataset is empty
        if len(X) == 0 or len(y) == 0:
            logger.error("Dataset is empty")
            return False
        
        logger.info(f"Dataset loaded successfully - {len(X)} samples, {len(np.unique(y))} classes")
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Initialize the sign recognizer
        recognizer = SignRecognizer(model_type="landmarks")
        
        # Train the model
        logger.info("Training the model...")
        accuracy = recognizer.train_landmark_model(X_train, y_train)
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Validate the model
        if X_test.shape[0] > 0:
            # Load the model again to ensure we're using the saved model
            validation_recognizer = SignRecognizer(model_type="landmarks")
            
            # Make predictions
            predictions = []
            for landmarks in X_test:
                result = validation_recognizer.predict_from_landmarks(landmarks)
                if result and result.get('letter'):
                    predictions.append(result.get('letter'))
                else:
                    predictions.append(None)
            
            # Calculate accuracy
            valid_predictions = [p for i, p in enumerate(predictions) if p is not None]
            valid_y_test = [y_test[i] for i, p in enumerate(predictions) if p is not None]
            
            if valid_predictions:
                val_accuracy = accuracy_score(valid_y_test, valid_predictions)
                logger.info(f"Validation accuracy: {val_accuracy:.4f}")
                logger.info("\nClassification Report:")
                logger.info(classification_report(valid_y_test, valid_predictions))
            else:
                logger.warning("No valid predictions for validation")
        
        return True
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train ASL Model from Dataset")
    parser.add_argument('--landmarks_path', default='./models/landmarks.npy', help='Path to the landmarks file')
    parser.add_argument('--labels_path', default='./models/labels.npy', help='Path to the labels file')
    parser.add_argument('--output_path', default='./models', help='Path to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Train the model
    if train_model(args.landmarks_path, args.labels_path, args.output_path, args.test_size, args.random_state):
        logger.info("Model trained successfully!")
        return 0
    
    logger.error("Failed to train model")
    return 1

if __name__ == "__main__":
    sys.exit(main()) 