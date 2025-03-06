"""
Visualization utilities for the ASL Finger Spelling Agent.
"""

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def draw_landmarks(image, hands_data, draw_text=True):
    """
    Draw hand landmarks on an image.
    
    Args:
        image (numpy.ndarray): Input image
        hands_data (list): List of detected hands with landmarks
        draw_text (bool): Whether to draw hand type text
        
    Returns:
        numpy.ndarray: Image with landmarks drawn
    """
    if image is None or hands_data is None:
        return image
    
    # Make a copy of the image
    img_copy = image.copy()
    
    # Colors for drawing
    landmark_color = (0, 255, 0)  # Green
    connection_color = (255, 0, 0)  # Blue
    text_color = (255, 255, 255)  # White
    
    # Landmarks and connections
    landmark_connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    for hand in hands_data:
        # Get landmarks
        landmarks = hand['landmarks']
        
        # Draw hand type
        if draw_text and 'type' in hand:
            hand_type = hand['type']
            confidence = hand.get('confidence', 0.0)
            
            # Get wrist position for text placement
            wrist = landmarks[0]
            text_pos = (wrist['pixel_x'], wrist['pixel_y'] - 20)
            
            cv2.putText(img_copy, f"{hand_type} ({confidence:.2f})", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Draw landmarks
        for idx, lm in enumerate(landmarks):
            cv2.circle(img_copy, (lm['pixel_x'], lm['pixel_y']), 5, landmark_color, -1)
            
            # Draw landmark index for debugging
            # cv2.putText(img_copy, str(idx), (lm['pixel_x'] + 5, lm['pixel_y'] - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Draw connections
        for connection in landmark_connections:
            start_idx, end_idx = connection
            
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (landmarks[start_idx]['pixel_x'], landmarks[start_idx]['pixel_y'])
                end_point = (landmarks[end_idx]['pixel_x'], landmarks[end_idx]['pixel_y'])
                
                cv2.line(img_copy, start_point, end_point, connection_color, 2)
    
    return img_copy

def draw_prediction_results(image, prediction_result, target_letter=None):
    """
    Draw prediction results on the image.
    
    Args:
        image (numpy.ndarray): Input image
        prediction_result (dict): Prediction results with letter and confidence
        target_letter (str, optional): Target letter for comparison
        
    Returns:
        numpy.ndarray: Image with prediction results drawn
    """
    if image is None or prediction_result is None:
        return image
    
    # Make a copy of the image
    img_copy = image.copy()
    
    # Get prediction info
    predicted_letter = prediction_result.get('letter')
    confidence = prediction_result.get('confidence', 0.0)
    
    if predicted_letter is None:
        # No prediction available
        text = "No hand detected"
        color = (120, 120, 120)  # Gray
    else:
        # Format the prediction text
        text = f"Predicted: {predicted_letter} ({confidence:.2f})"
        
        # Determine color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green (high confidence)
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow (medium confidence)
        else:
            color = (0, 0, 255)  # Red (low confidence)
        
        # If target letter is provided, check if prediction is correct
        if target_letter is not None:
            if predicted_letter == target_letter:
                text += f" ✓"
                color = (0, 255, 0)  # Green
            else:
                text += f" ✗ (Expected: {target_letter})"
                color = (0, 0, 255)  # Red
    
    # Draw text at the top of the image
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img_copy

def draw_alphabet_reference(size=(800, 600)):
    """
    Create an image with the ASL alphabet reference.
    
    Args:
        size (tuple): Size of the output image (width, height)
        
    Returns:
        numpy.ndarray: Image with ASL alphabet reference
    """
    # Create a blank image
    image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Define the grid
    rows, cols = 5, 6  # 5 rows, 6 columns (26 letters + space for J and Z which involve motion)
    cell_width = size[0] // cols
    cell_height = size[1] // rows
    
    # Define the alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Dictionary to note signs that involve motion
    motion_signs = {
        'J': "Trace a 'J' shape",
        'Z': "Trace a 'Z' shape"
    }
    
    for i, letter in enumerate(alphabet):
        # Calculate position
        row = i // cols
        col = i % cols
        
        # Calculate cell position
        x = col * cell_width
        y = row * cell_height
        
        # Draw cell border
        cv2.rectangle(image, (x, y), (x + cell_width, y + cell_height), (200, 200, 200), 1)
        
        # Draw letter
        letter_pos = (x + 10, y + 30)
        cv2.putText(image, letter, letter_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add note for motion signs
        if letter in motion_signs:
            note_pos = (x + 10, y + cell_height - 20)
            cv2.putText(image, motion_signs[letter], note_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Add title
    title = "ASL Fingerspelling Alphabet"
    title_pos = (size[0] // 2 - 200, 30)
    cv2.putText(image, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add note at the bottom
    note = "Note: J and Z involve motion and cannot be represented with a static image."
    note_pos = (20, size[1] - 20)
    cv2.putText(image, note, note_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def plot_confusion_matrix(confusion_matrix, classes, title='Confusion Matrix'):
    """
    Plot a confusion matrix.
    
    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix to plot
        classes (list): List of class names
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return plt.gcf()

def test_visualization():
    """
    Simple test function for the visualization utilities.
    """
    # Create a test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create dummy hand data
    hands_data = [{
        'type': 'Right',
        'confidence': 0.95,
        'landmarks': [
            {'x': 0.5, 'y': 0.5, 'z': 0.0, 'pixel_x': 320, 'pixel_y': 240},  # Wrist
            {'x': 0.6, 'y': 0.4, 'z': 0.0, 'pixel_x': 380, 'pixel_y': 200},  # Thumb
            # ... add more landmarks as needed
        ]
    }]
    
    # Draw landmarks
    result = draw_landmarks(image, hands_data)
    
    # Draw prediction results
    prediction = {'letter': 'A', 'confidence': 0.85}
    result = draw_prediction_results(result, prediction, target_letter='A')
    
    # Display result
    cv2.imshow("Visualization Test", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Create and display alphabet reference
    alphabet_ref = draw_alphabet_reference()
    cv2.imshow("ASL Alphabet Reference", alphabet_ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_visualization() 