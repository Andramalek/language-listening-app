"""
ASL Dataset Downloader and Processor

This script downloads a subset of the ASL Alphabet dataset from public sources,
extracts hand landmarks using MediaPipe, and creates a training dataset
for the SignRecognizer.
"""

import os
import sys
import logging
import argparse
import cv2
import numpy as np
import urllib.request
import zipfile
import shutil
from tqdm import tqdm
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Public dataset URLs (subsets of the ASL alphabet dataset with direct download links)
ASL_DATASET_URLS = {
    # GitHub hosted ASL samples
    'basic': 'https://github.com/loicmarie/sign-language-alphabet-recognizer/archive/master.zip'
}

class ASLDatasetProcessor:
    """Process ASL alphabet images to extract landmarks for training."""
    
    def __init__(self, dataset_path, output_path, dataset_type='basic'):
        """
        Initialize the dataset processor.
        
        Args:
            dataset_path (str): Path to download and extract the dataset
            output_path (str): Path to save the processed landmarks
            dataset_type (str): Type of dataset to use ('basic')
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.dataset_type = dataset_type
        self.dataset_url = ASL_DATASET_URLS.get(dataset_type)
        
        if not self.dataset_url:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def download_dataset(self):
        """Download and extract the dataset."""
        if self.dataset_type == 'basic':
            return self._download_basic_dataset()
    
    def _download_basic_dataset(self):
        """Download and extract the basic ASL dataset."""
        zip_path = os.path.join(self.dataset_path, 'asl_basic.zip')
        extract_path = os.path.join(self.dataset_path, 'basic')
        
        # Download the dataset if it doesn't exist
        if not os.path.exists(zip_path):
            logger.info(f"Downloading ASL dataset from {self.dataset_url}")
            try:
                urllib.request.urlretrieve(self.dataset_url, zip_path)
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                return False
        
        # Extract the dataset if it doesn't exist
        if not os.path.exists(extract_path):
            logger.info(f"Extracting ASL dataset to {extract_path}")
            os.makedirs(extract_path, exist_ok=True)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            except Exception as e:
                logger.error(f"Failed to extract dataset: {e}")
                return False
        
        return True
    
    def _find_images_path(self):
        """Find the path to the images in the extracted dataset."""
        if self.dataset_type == 'basic':
            # Path structure in the basic dataset
            extract_path = os.path.join(self.dataset_path, 'basic')
            # Navigate to the directory containing images
            root_dir = os.path.join(extract_path, 'sign-language-alphabet-recognizer-master')
            if os.path.exists(root_dir):
                dataset_dir = os.path.join(root_dir, 'dataset')
                if os.path.exists(dataset_dir):
                    return dataset_dir
            
            # Search for dataset directory
            for root, dirs, files in os.walk(extract_path):
                if 'dataset' in dirs:
                    return os.path.join(root, 'dataset')
        
        logger.error(f"Could not find images path in the extracted dataset")
        return None
    
    def extract_landmarks_from_image(self, image):
        """
        Extract normalized hand landmarks from an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray or None: Normalized hand landmarks, None if no hand detected
        """
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Extract landmarks
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract normalized (x, y, z) coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            landmarks_array = np.array(landmarks)
            
            # Normalize relative to wrist position
            wrist_pos = landmarks_array[0]
            normalized_landmarks = landmarks_array - wrist_pos
            
            # Flatten the array for ML input
            return normalized_landmarks.flatten()
        
        return None
    
    def process_dataset(self):
        """Process the dataset and extract landmarks."""
        # Find the path to the images
        images_path = self._find_images_path()
        if not images_path:
            return False
        
        logger.info(f"Processing images from {images_path}")
        
        # Initialize lists for landmarks and labels
        X = []
        y = []
        
        # Process each directory (letter)
        for letter_dir in os.listdir(images_path):
            letter_path = os.path.join(images_path, letter_dir)
            
            # Skip files and non-letter directories
            if not os.path.isdir(letter_path) or letter_dir not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                continue
            
            logger.info(f"Processing letter: {letter_dir}")
            
            # Process each image in the directory
            image_files = [f for f in os.listdir(letter_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for image_file in tqdm(image_files[:50], desc=f"Letter {letter_dir}"):  # Limit to 50 images per letter
                image_path = os.path.join(letter_path, image_file)
                
                # Read and process the image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to read image: {image_path}")
                    continue
                
                # Extract landmarks
                landmarks = self.extract_landmarks_from_image(image)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(letter_dir)
        
        # Save the landmarks and labels
        if X and y:
            logger.info(f"Saving {len(X)} processed samples")
            
            # Ensure the output directory exists
            os.makedirs(self.output_path, exist_ok=True)
            
            # Save as NumPy arrays
            X_array = np.array(X)
            y_array = np.array(y)
            
            np.save(os.path.join(self.output_path, 'landmarks.npy'), X_array)
            np.save(os.path.join(self.output_path, 'labels.npy'), y_array)
            
            logger.info(f"Dataset processed successfully!")
            logger.info(f"X shape: {X_array.shape}, y shape: {y_array.shape}")
            
            return True
        else:
            logger.error("No landmarks extracted from the dataset")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ASL Dataset Downloader and Processor")
    parser.add_argument('--dataset_path', default='./datasets', help='Path to download and extract the dataset')
    parser.add_argument('--output_path', default='./models', help='Path to save the processed landmarks')
    parser.add_argument('--dataset_type', default='basic', choices=['basic'], help='Type of dataset to use')
    args = parser.parse_args()
    
    # Initialize the processor
    processor = ASLDatasetProcessor(args.dataset_path, args.output_path, args.dataset_type)
    
    # Download and process the dataset
    if processor.download_dataset():
        if processor.process_dataset():
            logger.info("Dataset processed successfully!")
            return 0
    
    logger.error("Failed to process dataset")
    return 1

if __name__ == "__main__":
    sys.exit(main()) 