"""
Utility module for generating audio from text using Text-to-Speech (TTS).
"""

import os
import time
import logging
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioGenerator:
    """
    A class to generate audio from text using Text-to-Speech (TTS).
    """
    
    def __init__(self, output_dir="data/audio"):
        """
        Initialize the audio generator.
        
        Args:
            output_dir (str, optional): Directory to save audio files. Defaults to "data/audio".
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Audio output directory: {self.output_dir}")
    
    def generate_audio(self, text, lang="ja", filename=None, slow=False):
        """
        Generate audio from text using gTTS.
        
        Args:
            text (str): The text to convert to speech
            lang (str, optional): Language code. Defaults to "ja" for Japanese.
            filename (str, optional): Output filename. If None, a timestamp-based name is used.
            slow (bool, optional): Whether to speak slowly. Defaults to False.
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Generate a filename if not provided
            if not filename:
                timestamp = int(time.time())
                filename = f"tts_{timestamp}.mp3"
            
            # Ensure the filename has a .mp3 extension
            if not filename.endswith('.mp3'):
                filename += '.mp3'
            
            # Full path to the output file
            output_path = os.path.join(self.output_dir, filename)
            
            # Generate the audio
            logger.info(f"Generating audio for text: '{text[:50]}...'")
            tts = gTTS(text=text, lang=lang, slow=slow)
            tts.save(output_path)
            
            logger.info(f"Audio saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise
    
    def generate_question_audio(self, question_data, question_lang="en", answer_lang="ja"):
        """
        Generate audio for a question and its answer.
        
        Args:
            question_data (dict): Question data with 'question', 'options', and 'answer' keys
            question_lang (str, optional): Language code for the question. Defaults to "en".
            answer_lang (str, optional): Language code for the answer. Defaults to "ja".
            
        Returns:
            dict: Dictionary with paths to the generated audio files
        """
        results = {}
        
        try:
            # Generate audio for the question
            question_text = question_data.get('question', '')
            if question_text:
                question_filename = f"question_{int(time.time())}.mp3"
                question_path = self.generate_audio(question_text, lang=question_lang, filename=question_filename)
                results['question_audio'] = question_path
            
            # Generate audio for each option (if available)
            options = question_data.get('options', [])
            option_paths = []
            
            for i, option in enumerate(options):
                option_filename = f"option_{i+1}_{int(time.time())}.mp3"
                option_path = self.generate_audio(option, lang=answer_lang, filename=option_filename)
                option_paths.append(option_path)
            
            if option_paths:
                results['option_audios'] = option_paths
            
            # Generate audio for the explanation (if available)
            explanation = question_data.get('explanation', '')
            if explanation:
                explanation_filename = f"explanation_{int(time.time())}.mp3"
                explanation_path = self.generate_audio(explanation, lang=question_lang, filename=explanation_filename)
                results['explanation_audio'] = explanation_path
            
            return results
        except Exception as e:
            logger.error(f"Failed to generate audio for question: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        text = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) > 2 else "ja"
        
        try:
            generator = AudioGenerator()
            audio_path = generator.generate_audio(text, lang=lang)
            print(f"Generated audio saved to: {audio_path}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide a text to convert to speech as an argument") 