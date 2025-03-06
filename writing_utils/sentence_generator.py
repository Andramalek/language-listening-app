"""
Utility module for generating simple sentences and translations.
"""

import os
import random
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class SentenceGenerator:
    """
    A class to generate simple sentences in English and translate them to Japanese.
    """
    
    def __init__(self):
        """
        Initialize the sentence generator.
        """
        self.openai_client = None
        
        # Sentence templates for simple sentences
        self.templates = [
            "I see a {word}.",
            "There is a {word}.",
            "This is a {word}.",
            "I like the {word}.",
            "The {word} is nice.",
            "She has a {word}.",
            "He wants a {word}.",
            "Can you see the {word}?",
            "Where is the {word}?",
            "Please give me the {word}.",
        ]
        
        # Initialize OpenAI client if API key is available
        if OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")
    
    def generate_sentences(self, words, count_per_word=1):
        """
        Generate simple English sentences using the provided words.
        
        Args:
            words (list): List of words to include in sentences
            count_per_word (int): Number of sentences to generate per word
            
        Returns:
            list: List of generated English sentences
        """
        sentences = []
        
        for word in words:
            # Generate 'count_per_word' sentences for each word
            word_sentences = []
            
            for _ in range(count_per_word):
                # Choose a random template
                template = random.choice(self.templates)
                
                # Create the sentence by filling in the template
                sentence = template.format(word=word)
                
                word_sentences.append(sentence)
            
            # Add the sentences for this word to the main list
            sentences.extend(word_sentences)
        
        return sentences
    
    def translate_to_japanese(self, sentences):
        """
        Translate English sentences to Japanese.
        
        Args:
            sentences (list): List of English sentences to translate
            
        Returns:
            list: List of Japanese translations
        """
        # Check if OpenAI client is available
        if not self.openai_client:
            # If no API available, use placeholder translations
            logger.warning("OpenAI client not available, using placeholder translations")
            return [f"これは「{sentence}」の日本語訳です。" for sentence in sentences]
        
        try:
            translations = []
            
            # Process in batches to avoid rate limits
            batch_size = 5
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                # Combine sentences for a single API call
                prompt = "Translate the following English sentences to Japanese. Keep the translations simple and suitable for beginners:\n\n"
                
                for j, sentence in enumerate(batch):
                    prompt += f"{j+1}. {sentence}\n"
                
                # Call OpenAI API with new client format
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that translates English to Japanese. Provide only the translations, numbered according to the input."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent translations
                )
                
                # Process the response to extract translations with new format
                translation_text = response.choices[0].message.content.strip()
                
                # Parse the numbered translations
                batch_translations = []
                lines = translation_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line and any(line.startswith(f"{j+1}.") for j in range(len(batch))):
                        # Remove the number and any leading/trailing whitespace
                        translation = line.split('.', 1)[1].strip()
                        batch_translations.append(translation)
                
                # If we couldn't parse properly, use a fallback method
                if len(batch_translations) != len(batch):
                    logger.warning(f"Translation parsing issue, using fallback. Got {len(batch_translations)} translations for {len(batch)} sentences.")
                    batch_translations = [f"[Translation {j+1}]: {line}" for j, line in enumerate(lines[:len(batch)])]
                
                translations.extend(batch_translations)
            
            return translations
            
        except Exception as e:
            logger.error(f"Error translating sentences: {e}", exc_info=True)
            # Fallback to placeholder translations
            return [f"[Error translating]: {sentence}" for sentence in sentences] 