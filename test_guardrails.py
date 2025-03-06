"""
Test script for verifying guardrails implementation
"""
import time
import datetime
from utils.transcript_fetcher import extract_video_id, check_inappropriate_content, get_transcript_with_fallback
from utils.question_generator import QuestionGenerator

def test_url_validation():
    """Test the URL validation guardrail"""
    print("\n=== Testing URL Validation Guardrail ===")
    
    test_cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True),  # Valid YouTube URL
        ("https://youtu.be/dQw4w9WgXcQ", True),  # Valid shortened URL
        ("dQw4w9WgXcQ", True),  # Just the video ID
        ("https://youtube.com", False),  # Missing video ID
        ("https://youtu.be/", False),  # Missing video ID
        ("abcdefg", False),  # Invalid string
        ("https://vimeo.com/12345", False),  # Wrong platform
        ("", False),  # Empty string
    ]
    
    for url, should_pass in test_cases:
        try:
            video_id = extract_video_id(url)
            result = f"✓ Extracted ID: {video_id}"
            if not should_pass:
                result = f"✗ Should have failed but passed: {video_id}"
        except ValueError as e:
            result = f"✗ Error: {e}"
            if should_pass:
                result = f"✗ Should have passed but failed: {e}"
            else:
                result = f"✓ Failed as expected: {e}"
        
        print(f"URL: {url} - {result}")

def test_content_filtering():
    """Test the content filtering guardrail"""
    print("\n=== Testing Content Filtering Guardrail ===")
    
    # Create some mock transcripts
    clean_transcript = [{"text": "This is a clean educational video about language learning."}]
    
    inappropriate_transcript = [{"text": "This video contains explicit content and nsfw material."}]
    
    profanity_transcript = [
        {"text": "This video has fuck too much profanity."},
        {"text": "It has shit many bad words."},
        {"text": "And it's damn not appropriate for education."},
        {"text": "This is ass not good."},
        {"text": "One more fuck bad word."},
        {"text": "Last shit bad word."}
    ]
    
    # Test each transcript
    print(f"Clean transcript - Inappropriate: {check_inappropriate_content(clean_transcript)}")
    print(f"Inappropriate transcript - Inappropriate: {check_inappropriate_content(inappropriate_transcript)}")
    print(f"Profanity transcript - Inappropriate: {check_inappropriate_content(profanity_transcript)}")

def test_rate_limiting():
    """Test the API rate limiting guardrail"""
    print("\n=== Testing API Rate Limiting Guardrail ===")
    
    # Create a question generator
    generator = QuestionGenerator()
    
    # Override rate limits for testing
    generator.max_requests_per_minute = 3
    
    # Make multiple requests in quick succession
    for i in range(5):
        try:
            print(f"Request {i+1}: ", end="")
            # Just check the rate limits without making an actual API call
            generator._check_rate_limits()
            # Record this request as if it was made
            generator.request_timestamps.append(datetime.datetime.now())
            generator.daily_request_count += 1
            print("✓ Passed")
        except RuntimeError as e:
            print(f"✗ Failed as expected: {e}")
            # Wait a bit to let the rate limit reset
            if i < 4:  # Only wait if not the last iteration
                print("Waiting 2 seconds...")
                time.sleep(2)

if __name__ == "__main__":
    # Run the tests
    test_url_validation()
    test_content_filtering()
    test_rate_limiting() 