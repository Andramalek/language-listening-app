"""
Utility module for generating and storing vector embeddings.
"""

import os
import sqlite3
import pickle
import logging
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default OpenAI API key and models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

class VectorStore:
    """
    A class to manage vector embeddings and storage using SQLite.
    """
    
    def __init__(self, db_path="data/vector_store.db"):
        """
        Initialize the vector store.
        
        Args:
            db_path (str, optional): Path to the SQLite database. Defaults to "data/vector_store.db".
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._setup_db()
        self.openai_client = None
        
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
    
    def _ensure_db_dir(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _setup_db(self):
        """Set up the SQLite database with necessary tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create the vectors table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    text TEXT,
                    start_time REAL,
                    duration REAL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create an index on video_id for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON vectors(video_id)")
            
            conn.commit()
            conn.close()
            logger.info(f"Database setup complete at {self.db_path}")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    def get_embedding(self, text):
        """
        Generate an embedding for the given text using OpenAI's API.
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embedding vector
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Check your API key.")
        
        try:
            response = self.openai_client.Embedding.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def store_transcript_embeddings(self, transcript, video_id):
        """
        Generate and store embeddings for transcript entries.
        
        Args:
            transcript (list): List of transcript entries with 'text', 'start', and 'duration' keys
            video_id (str): The YouTube video ID
            
        Returns:
            int: Number of entries stored
        """
        if not transcript:
            logger.warning("Empty transcript provided")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for entry in transcript:
            try:
                text = entry['text']
                embedding = self.get_embedding(text)
                
                cursor.execute(
                    "INSERT INTO vectors (video_id, text, start_time, duration, embedding) VALUES (?, ?, ?, ?, ?)",
                    (video_id, text, entry.get('start', 0), entry.get('duration', 0), pickle.dumps(embedding))
                )
                count += 1
                
                if count % 10 == 0:
                    logger.info(f"Stored {count} embeddings so far...")
            except Exception as e:
                logger.error(f"Failed to store embedding for entry: {entry}. Error: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully stored {count} transcript embeddings for video {video_id}")
        return count
    
    def search_similar(self, query_text, video_id=None, limit=5):
        """
        Search for transcript entries similar to the query text.
        
        Args:
            query_text (str): The query text to search for
            video_id (str, optional): Filter by video ID. Defaults to None.
            limit (int, optional): Maximum number of results. Defaults to 5.
            
        Returns:
            list: List of dictionaries with 'text', 'start_time', 'video_id', and 'similarity' keys
        """
        try:
            # Generate embedding for the query text
            query_embedding = self.get_embedding(query_text)
            
            # Fetch all vectors from the database (with video_id filter if provided)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if video_id:
                cursor.execute("SELECT id, video_id, text, start_time, duration, embedding FROM vectors WHERE video_id = ?", (video_id,))
            else:
                cursor.execute("SELECT id, video_id, text, start_time, duration, embedding FROM vectors")
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                logger.warning("No vectors found in the database")
                return []
            
            # Calculate similarities and prepare results
            similarities = []
            for row_id, vid_id, text, start, duration, embedding_blob in results:
                embedding = pickle.loads(embedding_blob)
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                similarities.append({
                    'id': row_id,
                    'video_id': vid_id,
                    'text': text,
                    'start_time': start,
                    'duration': duration,
                    'similarity': similarity
                })
            
            # Sort by similarity (highest first) and limit results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:limit]
            
            logger.info(f"Found {len(top_results)} similar entries for query: '{query_text}'")
            return top_results
        except Exception as e:
            logger.error(f"Error searching for similar entries: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        text_to_embed = sys.argv[1]
        try:
            vs = VectorStore()
            embedding = vs.get_embedding(text_to_embed)
            print(f"Generated embedding with {len(embedding)} dimensions")
            print(f"First 5 dimensions: {embedding[:5]}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide a text to embed as an argument") 