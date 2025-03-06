"""
Utility module for generating listening comprehension questions.
"""

import os
import logging
import time
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default OpenAI API key and models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

class QuestionGenerator:
    """
    A class to generate listening comprehension questions using LLMs.
    """
    
    def __init__(self):
        """
        Initialize the question generator.
        """
        self.openai_client = None
        
        # API usage guardrails
        self.max_requests_per_minute = int(os.getenv("MAX_API_REQUESTS_PER_MINUTE", "20"))
        self.max_requests_per_day = int(os.getenv("MAX_API_REQUESTS_PER_DAY", "1000"))
        self.request_timestamps = []
        self.daily_request_count = 0
        self.last_day_reset = datetime.now()
        
        if OPENAI_API_KEY:
            try:
                import openai
                openai.api_key = OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.warning("OpenAI package not installed. Run 'pip install openai'")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")
    
    def _check_rate_limits(self):
        """
        Check if we're within rate limits for API calls.
        
        Raises:
            RuntimeError: If rate limits would be exceeded
        """
        current_time = datetime.now()
        
        # Reset daily counter if it's a new day
        if (current_time - self.last_day_reset).days > 0:
            self.daily_request_count = 0
            self.last_day_reset = current_time
        
        # Check daily limit
        if self.daily_request_count >= self.max_requests_per_day:
            raise RuntimeError(f"Daily API request limit reached ({self.max_requests_per_day}). Please try again tomorrow.")
        
        # Clean up old timestamps (older than 1 minute)
        one_minute_ago = current_time - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > one_minute_ago]
        
        # Check per-minute limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest).seconds
            raise RuntimeError(f"Rate limit reached. Please wait {wait_time} seconds before trying again.")
    
    def generate_question(self, text, difficulty="intermediate", question_type="comprehension"):
        """
        Generate a listening comprehension question based on the text.
        
        Args:
            text (str): The text to generate a question from
            difficulty (str, optional): Difficulty level (beginner, intermediate, advanced). 
                                        Defaults to "intermediate".
            question_type (str, optional): Type of question (comprehension, vocabulary, grammar). 
                                          Defaults to "comprehension".
        
        Returns:
            dict: Dictionary with 'question', 'options' (for multiple choice), and 'answer' keys
            
        Raises:
            ValueError: If OpenAI client is not initialized
            RuntimeError: If rate limits are exceeded
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Check your API key.")
            
        # Apply rate limiting guardrail
        self._check_rate_limits()
        
        # Record this request
        self.request_timestamps.append(datetime.now())
        self.daily_request_count += 1
        
        # Create a prompt for the language model
        system_prompt = f"""
        You are a Japanese language teacher creating listening comprehension questions for English speakers learning Japanese.
        
        Generate a question based on the following Japanese text. The student's level is {difficulty}.
        
        Guidelines:
        - For beginners: Focus on basic vocabulary and simple sentence structures
        - For intermediate: Include more complex grammar and vocabulary
        - For advanced: Include nuanced understanding and cultural context
        
        The question should be a {question_type} question.
        - For comprehension: Test understanding of the main ideas
        - For vocabulary: Test understanding of specific words or phrases
        - For grammar: Test understanding of grammatical structures
        
        Return your response in JSON format with these fields:
        - question: The question in English
        - options: Array of 4 possible answers (for multiple choice)
        - answer: The correct answer (must be one of the options)
        - explanation: Brief explanation of why the answer is correct (in English)
        """
        
        try:
            response = self.openai_client.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.7
            )
            
            # Extract the response
            result = response.choices[0].message.content
            logger.info(f"Generated question for text: '{text[:50]}...'")
            
            # Parse the JSON response
            try:
                import json
                result_json = json.loads(result)
                return result_json
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                return {
                    "question": result,
                    "options": [],
                    "answer": "",
                    "explanation": "Failed to parse structured response"
                }
            
        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            raise
    
    def generate_questions_batch(self, texts, count_per_text=1, **kwargs):
        """
        Generate multiple questions from a list of texts.
        
        Args:
            texts (list): List of texts to generate questions from
            count_per_text (int, optional): Number of questions per text. Defaults to 1.
            **kwargs: Additional parameters to pass to generate_question
            
        Returns:
            list: List of question dictionaries
        """
        all_questions = []
        
        for i, text in enumerate(texts):
            logger.info(f"Generating questions for text {i+1}/{len(texts)}")
            for j in range(count_per_text):
                try:
                    question = self.generate_question(text, **kwargs)
                    all_questions.append(question)
                except Exception as e:
                    logger.error(f"Failed to generate question for text {i+1}, attempt {j+1}: {e}")
        
        logger.info(f"Generated {len(all_questions)} questions in total")
        return all_questions

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        text = sys.argv[1]
        try:
            generator = QuestionGenerator()
            question = generator.generate_question(text)
            print(f"Question: {question['question']}")
            if 'options' in question and question['options']:
                for i, option in enumerate(question['options']):
                    print(f"  {chr(65+i)}. {option}")
            print(f"Answer: {question['answer']}")
            if 'explanation' in question:
                print(f"Explanation: {question['explanation']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide a text to generate a question from as an argument") 