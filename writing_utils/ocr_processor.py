"""
Utility module for processing OCR on handwritten Japanese text.
"""

import os
import io
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OCRProcessor:
    """
    A class to process OCR on handwritten Japanese text.
    Uses MangaOCR or OpenAI's GPT-4V for image analysis.
    """
    
    def __init__(self, use_mangaocr=True):
        """
        Initialize the OCR processor.
        
        Args:
            use_mangaocr (bool): Whether to use MangaOCR (True) or OpenAI Vision (False)
        """
        self.use_mangaocr = use_mangaocr
        self.manga_ocr = None
        self.openai_client = None
        
        # Initialize the appropriate OCR engine
        if use_mangaocr:
            try:
                # We'll import MangaOCR only if it's being used
                # to avoid unnecessary dependencies
                from manga_ocr import MangaOcr
                self.manga_ocr = MangaOcr()
                logger.info("MangaOCR initialized successfully")
            except ImportError:
                logger.warning("manga-ocr package not installed. Run 'pip install manga-ocr'")
                self.use_mangaocr = False
            except Exception as e:
                logger.error(f"Failed to initialize MangaOCR: {e}")
                self.use_mangaocr = False
        
        # If MangaOCR isn't available or not selected, initialize OpenAI
        if not self.use_mangaocr:
            if OPENAI_API_KEY:
                try:
                    # Initialize with new OpenAI client
                    self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    logger.info("OpenAI client initialized for Vision OCR")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
            else:
                logger.warning("OPENAI_API_KEY not found in environment variables")
    
    def process_image(self, image_data, expected_text=None):
        """
        Process an image containing handwritten Japanese text.
        
        Args:
            image_data: Image data (can be bytes, file-like object, or path)
            expected_text (str, optional): The expected Japanese text for comparison
            
        Returns:
            dict: Dictionary with 'text' (detected text), 'confidence' (confidence score),
                  and 'feedback' (comparison with expected text if provided)
        """
        if self.use_mangaocr and self.manga_ocr:
            return self._process_with_mangaocr(image_data, expected_text)
        elif self.openai_client:
            return self._process_with_openai_vision(image_data, expected_text)
        else:
            error_msg = "No OCR engine available. Please install manga-ocr or provide an OpenAI API key."
            logger.error(error_msg)
            return {"text": "", "confidence": 0, "feedback": error_msg, "error": True}
    
    def _process_with_mangaocr(self, image_data, expected_text):
        """
        Process an image using MangaOCR.
        """
        try:
            # If image_data is bytes or file-like object, save to a temporary file
            if isinstance(image_data, (bytes, io.IOBase)):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    if isinstance(image_data, bytes):
                        temp_file.write(image_data)
                    else:  # file-like object
                        temp_file.write(image_data.read())
                
                # Process the temporary file
                text = self.manga_ocr(temp_path)
                
                # Clean up
                os.unlink(temp_path)
            else:
                # Assume image_data is a path
                text = self.manga_ocr(str(image_data))
            
            # Generate feedback if expected text is provided
            feedback = self._generate_feedback(text, expected_text) if expected_text else ""
            
            result = {
                "text": text,
                "confidence": 0.8,  # MangaOCR doesn't provide confidence scores, so we use a placeholder
                "feedback": feedback
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image with MangaOCR: {e}")
            return {"text": "", "confidence": 0, "feedback": f"Error: {str(e)}", "error": True}
    
    def _process_with_openai_vision(self, image_data, expected_text):
        """
        Process an image using OpenAI's GPT-4o model with vision capabilities.
        """
        try:
            # Convert image data to base64 if needed
            if isinstance(image_data, (str, Path)):
                # It's a file path
                logger.info(f"Processing image from path: {image_data}")
                with open(image_data, "rb") as f:
                    image_bytes = f.read()
            elif isinstance(image_data, io.IOBase):
                # It's a file-like object
                logger.info("Processing image from file-like object")
                image_bytes = image_data.read()
            else:
                # Assume it's already bytes
                logger.info("Processing image from bytes")
                image_bytes = image_data
            
            # Log image size for debugging
            logger.info(f"Image size: {len(image_bytes)} bytes")
            
            # Encode image as base64 for logging purposes only
            import base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            logger.info(f"Base64 encoding successful, length: {len(base64_image)}")
            
            # Prepare the prompt
            prompt = "This is a handwritten Japanese text. Please recognize and transcribe the text exactly as written."
            
            if expected_text:
                prompt += f" For reference, the expected text is: {expected_text}"
            
            logger.info(f"Calling OpenAI API with prompt: {prompt}")
            
            # Call OpenAI API with new client format and updated model
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Updated to use gpt-4o instead of deprecated gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            logger.info("OpenAI API call successful")
            
            # Extract the recognized text from the new response format
            result_text = response.choices[0].message.content.strip()
            logger.info(f"Received response: {result_text[:100]}...")
            
            # Process the result to extract the actual Japanese text
            lines = result_text.split("\n")
            japanese_text = ""
            
            for line in lines:
                # Look for Japanese characters in the line
                if any(ord(c) > 127 for c in line):  # Non-ASCII characters
                    japanese_text = line
                    break
            
            # If no clear Japanese text was found, use the full response
            if not japanese_text:
                japanese_text = result_text
            
            # Generate feedback if expected text is provided
            feedback = self._generate_feedback(japanese_text, expected_text) if expected_text else ""
            
            return {
                "text": japanese_text,
                "confidence": 0.9,  # Placeholder, as GPT doesn't provide explicit confidence scores
                "feedback": feedback,
                "full_response": result_text  # Include the full response for debugging
            }
            
        except Exception as e:
            logger.error(f"Error processing image with OpenAI Vision: {e}", exc_info=True)
            return {"text": "", "confidence": 0, "feedback": f"Error: {str(e)}", "error": True}
    
    def _generate_feedback(self, detected_text, expected_text):
        """
        Compare detected text with expected text and generate feedback.
        
        Args:
            detected_text (str): The text detected by OCR
            expected_text (str): The expected Japanese text
            
        Returns:
            str: Feedback on the handwriting
        """
        if not detected_text or not expected_text:
            return "Unable to provide feedback without both detected and expected text."
        
        # Remove spaces and normalize for comparison
        detected_clean = ''.join(detected_text.split())
        expected_clean = ''.join(expected_text.split())
        
        # Calculate a simple character-level accuracy
        correct_chars = sum(1 for a, b in zip(detected_clean, expected_clean) if a == b)
        max_length = max(len(detected_clean), len(expected_clean))
        accuracy = correct_chars / max_length if max_length > 0 else 0
        
        # Generate feedback based on accuracy
        if accuracy > 0.9:
            return "Excellent! Your handwriting is very clear and accurate."
        elif accuracy > 0.7:
            return "Good job! Most characters are correct. Keep practicing for further improvement."
        elif accuracy > 0.5:
            return "Fair attempt. Some characters are recognized correctly, but there's room for improvement."
        else:
            return "Keep practicing. Try to write more clearly and check your character formation." 