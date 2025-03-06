"""
Utility module for fetching transcripts from YouTube videos.
"""

import re
import os
from youtube_transcript_api import YouTubeTranscriptApi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of potential inappropriate content keywords
INAPPROPRIATE_KEYWORDS = [
    'explicit', 'nsfw', 'adult content', 'violence', 'graphic',
    # Add more keywords as needed
]

# Content filtering settings
ENABLE_CONTENT_FILTER = os.getenv("ENABLE_CONTENT_FILTER", "true").lower() == "true"

def extract_video_id(youtube_url):
    """
    Extract the video ID from a YouTube URL.
    
    Args:
        youtube_url (str): The YouTube video URL
        
    Returns:
        str: The YouTube video ID
        
    Raises:
        ValueError: If the URL is not a valid YouTube URL
    """
    # Input validation guardrail
    if not youtube_url or not isinstance(youtube_url, str):
        raise ValueError("YouTube URL must be a non-empty string")
    
    # URL format validation
    if "youtu.be" in youtube_url:
        video_id = youtube_url.split("/")[-1].split("?")[0]
    elif "youtube.com" in youtube_url:
        if "v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "embed" in youtube_url:
            video_id = youtube_url.split("/")[-1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format. Expected 'youtube.com/watch?v=VIDEO_ID' or similar format")
    else:
        # Check if it might be just a video ID (11-character string)
        if len(youtube_url) == 11 and all(c in '-_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' for c in youtube_url):
            video_id = youtube_url
        else:
            raise ValueError("Invalid YouTube URL. Must contain 'youtube.com' or 'youtu.be'")
    
    # Validate video ID format
    if not video_id or len(video_id) != 11:
        raise ValueError(f"Invalid YouTube video ID extracted: {video_id}")
        
    return video_id

def get_transcript(video_id_or_url, languages=None):
    """
    Fetch the transcript for a YouTube video.
    
    Args:
        video_id_or_url (str): YouTube video ID or URL
        languages (list, optional): List of language codes to fetch. Defaults to ['ja', 'en'].
        
    Returns:
        list: List of transcript entries with 'text', 'start', and 'duration' keys
    """
    # Set default languages if not provided
    if languages is None:
        languages = ['ja', 'en']
    
    # Extract video ID if a URL was provided
    video_id = extract_video_id(video_id_or_url)
    
    try:
        logger.info(f"Fetching transcript for video ID: {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        logger.info(f"Successfully fetched transcript with {len(transcript)} entries")
        return transcript
    except Exception as e:
        logger.error(f"Error fetching transcript: {e}")
        raise

def format_transcript(transcript, include_timestamps=False):
    """
    Format the transcript entries as a list of text chunks.
    
    Args:
        transcript (list): List of transcript entries
        include_timestamps (bool, optional): Whether to include timestamps. Defaults to False.
        
    Returns:
        list: List of formatted text chunks
    """
    if include_timestamps:
        return [f"{entry['start']:.2f}s: {entry['text']}" for entry in transcript]
    else:
        return [entry['text'] for entry in transcript]

def get_transcript_with_fallback(video_id_or_url, languages=None):
    """
    Attempt to fetch transcript with fallback options.
    
    Args:
        video_id_or_url (str): YouTube video ID or URL
        languages (list, optional): List of language codes to fetch. Defaults to ['ja', 'en'].
        
    Returns:
        list: List of transcript entries
        
    Raises:
        ValueError: If the transcript couldn't be fetched or contains inappropriate content
    """
    # Extract video ID if a URL was provided
    video_id = extract_video_id(video_id_or_url)
    
    # Set default languages if not provided
    if languages is None:
        languages = ['ja', 'en']
    
    try:
        # Try with primary languages
        return get_transcript(video_id, languages=languages)
    except Exception as e:
        logger.warning(f"Failed to get transcript with primary languages {languages}: {e}")
        if languages != ['en']:
            # Fallback to English if not already tried
            try:
                logger.info("Trying fallback to English transcript")
                transcript = get_transcript(video_id, languages=['en'])
                
                # Content filtering guardrail
                if ENABLE_CONTENT_FILTER and check_inappropriate_content(transcript):
                    raise ValueError("This video appears to contain inappropriate content and has been filtered.")
                
                return transcript
            except Exception as e2:
                logger.error(f"Failed to get English transcript: {e2}")
        
        # If all attempts failed
        raise ValueError(f"Could not fetch transcript for video {video_id}. Please try a different video.")

def check_inappropriate_content(transcript):
    """
    Check if the transcript contains potentially inappropriate content.
    
    Args:
        transcript (list): List of transcript entries
        
    Returns:
        bool: True if inappropriate content is detected, False otherwise
    """
    if not ENABLE_CONTENT_FILTER:
        return False
        
    combined_text = " ".join([entry['text'].lower() for entry in transcript])
    
    # Check for inappropriate keywords
    for keyword in INAPPROPRIATE_KEYWORDS:
        if keyword.lower() in combined_text:
            logger.warning(f"Inappropriate content detected: '{keyword}'")
            return True
    
    # Check for excessive profanity (simple heuristic)
    profanity_count = sum(1 for word in combined_text.split() 
                         if word in ['fuck', 'shit', 'ass', 'damn'])
    if profanity_count > 5:
        logger.warning(f"Excessive profanity detected: {profanity_count} instances")
        return True
    
    return False

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        try:
            transcript = get_transcript_with_fallback(video_url)
            for entry in transcript[:5]:  # Print first 5 entries
                print(f"{entry['start']:.2f}s - {entry['text']}")
            print(f"... and {len(transcript) - 5} more entries")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide a YouTube video URL or ID as an argument") 