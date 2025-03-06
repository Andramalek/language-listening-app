"""
Language Listening App - Main Streamlit Application

A language learning application for English speakers learning Japanese,
using YouTube video transcripts to generate listening comprehension exercises.
"""

import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv

# Import utility modules
from utils import (
    get_transcript_with_fallback,
    extract_video_id,
    format_transcript,
    VectorStore,
    QuestionGenerator,
    AudioGenerator
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Language Listening App",
    page_icon="🇯🇵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure application limits and guardrails
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20"))
MAX_AUDIO_FILES_PER_SESSION = int(os.getenv("MAX_AUDIO_FILES_PER_SESSION", "50"))
MAX_SEARCH_QUERIES_PER_MINUTE = int(os.getenv("MAX_SEARCH_QUERIES_PER_MINUTE", "10"))

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'total_answered' not in st.session_state:
    st.session_state.total_answered = 0
if 'audio_paths' not in st.session_state:
    st.session_state.audio_paths = {}
# Add guardrail session state variables
if 'last_error_time' not in st.session_state:
    st.session_state.last_error_time = None
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'search_timestamps' not in st.session_state:
    st.session_state.search_timestamps = []

# Helper functions
def process_youtube_url(youtube_url):
    """Process a YouTube URL, fetch transcript, and store in vector database."""
    try:
        # Reset error count if more than 1 hour has passed
        if st.session_state.last_error_time and (time.time() - st.session_state.last_error_time > 3600):
            st.session_state.error_count = 0
        
        # Guardrail: If too many errors in a short time, suggest user takes a break
        if st.session_state.error_count >= 5:
            st.error("Multiple errors detected. Please take a break and try again later, or contact support if issues persist.")
            return None
        
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        st.session_state.video_id = video_id
        
        with st.spinner("Fetching transcript from YouTube..."):
            # Fetch transcript
            transcript = get_transcript_with_fallback(video_id)
            
            if not transcript:
                raise ValueError("No transcript available for this video.")
                
            # Guardrail: Check if transcript is long enough to be useful
            if len(transcript) < 10:
                st.warning("This video has a very short transcript. You may want to try a longer video for better learning content.")
            
            st.session_state.transcript = transcript
            
            # Store in vector database (in the background)
            vector_store = VectorStore()
            vector_store.store_transcript_embeddings(transcript, video_id)
            
            return transcript
    except ValueError as e:
        st.session_state.last_error_time = time.time()
        st.session_state.error_count += 1
        st.error(f"{str(e)} Please check the URL and try again.")
        logger.error(f"Error processing YouTube URL: {e}")
        return None
    except Exception as e:
        st.session_state.last_error_time = time.time()
        st.session_state.error_count += 1
        st.error(f"Error processing YouTube video: {e}")
        logger.error(f"Unexpected error processing YouTube URL: {e}", exc_info=True)
        return None

def generate_questions_from_transcript(transcript, count=5, difficulty="intermediate"):
    """Generate questions from the transcript."""
    try:
        # Guardrail: Limit number of questions per session
        if len(st.session_state.questions) >= MAX_QUESTIONS_PER_SESSION:
            st.warning(f"You've reached the maximum of {MAX_QUESTIONS_PER_SESSION} questions per session. Please refresh the page to start a new session.")
            return []
        
        with st.spinner("Generating questions..."):
            # Format transcript for question generation
            formatted_texts = format_transcript(transcript)
            
            # Guardrail: Ensure we're not asking for too many questions
            safe_count = min(count, 10)  # Maximum 10 questions per request
            
            # Generate questions
            generator = QuestionGenerator()
            questions = generator.generate_questions_batch(
                formatted_texts, 
                count_per_text=1, 
                difficulty=difficulty
            )
            
            # Limit to the requested count
            return questions[:safe_count]
    except RuntimeError as e:
        # Handle rate limiting errors explicitly
        if "rate limit" in str(e).lower():
            st.warning(f"API Rate limit reached: {e}")
            logger.warning(f"Rate limit error: {e}")
        else:
            st.error(f"Error generating questions: {e}")
            logger.error(f"Runtime error generating questions: {e}")
        return []
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        logger.error(f"Error generating questions: {e}", exc_info=True)
        return []

def search_transcript_by_topic(topic, video_id=None):
    """Search the transcript for content related to a topic."""
    try:
        # Guardrail: Rate limiting for search queries
        current_time = time.time()
        # Clean old timestamps (older than 1 minute)
        st.session_state.search_timestamps = [ts for ts in st.session_state.search_timestamps 
                                             if current_time - ts < 60]
        # Check if too many searches in the last minute
        if len(st.session_state.search_timestamps) >= MAX_SEARCH_QUERIES_PER_MINUTE:
            st.warning(f"Search rate limit reached. Please wait a moment before searching again.")
            return []
            
        # Record this search
        st.session_state.search_timestamps.append(current_time)
        
        with st.spinner("Searching for relevant content..."):
            # Search vector database
            vector_store = VectorStore()
            results = vector_store.search_similar(topic, video_id=video_id, limit=5)
            return results
    except Exception as e:
        st.error(f"Error searching transcript: {e}")
        logger.error(f"Error searching transcript: {e}", exc_info=True)
        return []

def generate_audio_for_question(question_data):
    """Generate audio for a question and its options."""
    try:
        # Guardrail: Limit number of audio files per session
        current_audio_count = len(st.session_state.audio_paths.keys())
        if current_audio_count >= MAX_AUDIO_FILES_PER_SESSION:
            st.warning(f"You've reached the maximum of {MAX_AUDIO_FILES_PER_SESSION} audio files per session. Please refresh to start a new session.")
            return {}
            
        with st.spinner("Generating audio..."):
            generator = AudioGenerator()
            audio_paths = generator.generate_question_audio(
                question_data,
                question_lang="en",
                answer_lang="ja"
            )
            return audio_paths
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        logger.error(f"Error generating audio: {e}", exc_info=True)
        return {}

def next_question():
    """Move to the next question."""
    if st.session_state.current_question < len(st.session_state.questions) - 1:
        st.session_state.current_question += 1
        st.session_state.show_answer = False
    else:
        st.info("You've completed all questions!")

def show_answer():
    """Show the answer for the current question."""
    st.session_state.show_answer = True

def check_answer(selected_option, correct_answer):
    """Check if the selected answer is correct."""
    if selected_option == correct_answer:
        st.session_state.score += 1
        st.success("Correct! 🎉")
    else:
        st.error("Incorrect. Try again.")
    st.session_state.total_answered += 1
    st.session_state.show_answer = True

# Main application
st.title("🇯🇵 Language Listening App")
st.subheader("Learn Japanese through listening exercises")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # YouTube URL input
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    if st.button("Fetch Transcript"):
        if youtube_url:
            transcript = process_youtube_url(youtube_url)
            if transcript:
                st.success(f"Successfully fetched transcript with {len(transcript)} entries.")
                st.session_state.questions = []  # Reset questions
                st.session_state.current_question = 0
        else:
            st.warning("Please enter a YouTube URL.")
    
    # Question generation settings
    st.subheader("Question Settings")
    
    question_count = st.slider(
        "Number of Questions",
        min_value=1,
        max_value=10,
        value=5
    )
    
    difficulty = st.select_slider(
        "Difficulty Level",
        options=["beginner", "intermediate", "advanced"],
        value="intermediate"
    )
    
    if st.button("Generate Questions"):
        if st.session_state.transcript:
            questions = generate_questions_from_transcript(
                st.session_state.transcript,
                count=question_count,
                difficulty=difficulty
            )
            st.session_state.questions = questions
            st.session_state.current_question = 0
            st.session_state.show_answer = False
            st.session_state.score = 0
            st.session_state.total_answered = 0
            st.session_state.audio_paths = {}
        else:
            st.warning("Please fetch a transcript first.")
    
    # Topic search
    st.subheader("Search by Topic")
    
    topic = st.text_input(
        "Topic",
        placeholder="Enter a topic to search for..."
    )
    
    if st.button("Search"):
        if topic:
            results = search_transcript_by_topic(topic, st.session_state.video_id)
            
            if results:
                st.subheader("Search Results")
                for result in results:
                    with st.expander(f"{result['text'][:50]}..."):
                        st.write(f"Text: {result['text']}")
                        st.write(f"Timestamp: {result['start_time']:.2f}s")
                        st.write(f"Similarity: {result['similarity']:.4f}")
            else:
                st.info("No results found.")
        else:
            st.warning("Please enter a topic to search for.")

# Main content
if st.session_state.questions and len(st.session_state.questions) > 0:
    # Display question
    current_q = st.session_state.questions[st.session_state.current_question]
    
    # Generate audio if not already done
    if st.session_state.current_question not in st.session_state.audio_paths:
        audio_paths = generate_audio_for_question(current_q)
        st.session_state.audio_paths[st.session_state.current_question] = audio_paths
    
    # Display question count and score
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Question {st.session_state.current_question + 1} of {len(st.session_state.questions)}")
    with col2:
        if st.session_state.total_answered > 0:
            st.write(f"Score: {st.session_state.score}/{st.session_state.total_answered} ({st.session_state.score/st.session_state.total_answered*100:.1f}%)")
    
    # Question and audio
    st.subheader(current_q.get('question', 'Question not available'))
    
    # Play question audio
    audio_paths = st.session_state.audio_paths.get(st.session_state.current_question, {})
    
    if 'question_audio' in audio_paths:
        st.audio(audio_paths['question_audio'], format='audio/mp3')
    
    # Options
    options = current_q.get('options', [])
    correct_answer = current_q.get('answer', '')
    
    if options:
        selected_option = None
        option_cols = st.columns(len(options))
        
        for i, (col, option) in enumerate(zip(option_cols, options)):
            with col:
                option_label = chr(65 + i)  # A, B, C, D...
                
                # Check if option audio exists
                option_audio = audio_paths.get('option_audios', [])[i] if i < len(audio_paths.get('option_audios', [])) else None
                
                if option_audio:
                    st.audio(option_audio, format='audio/mp3')
                
                if st.button(f"{option_label}. {option}", key=f"option_{i}"):
                    if not st.session_state.show_answer:
                        check_answer(option, correct_answer)
    
    # Show answer button
    if not st.session_state.show_answer:
        st.button("Show Answer", on_click=show_answer)
    
    # Display explanation if answer is shown
    if st.session_state.show_answer:
        st.markdown("### Answer")
        st.success(f"Correct answer: {correct_answer}")
        
        explanation = current_q.get('explanation', '')
        if explanation:
            st.markdown("### Explanation")
            st.info(explanation)
            
            # Play explanation audio if available
            explanation_audio = audio_paths.get('explanation_audio')
            if explanation_audio:
                st.audio(explanation_audio, format='audio/mp3')
    
    # Next question button
    if st.button("Next Question"):
        next_question()
        st.experimental_rerun()

else:
    # Instructions when no questions are available
    st.info("👈 Enter a YouTube URL in the sidebar to get started.")
    
    with st.expander("How to use this app"):
        st.markdown("""
        1. Paste a YouTube URL with Japanese content in the sidebar
        2. Click 'Fetch Transcript' to load the transcript
        3. Set the number of questions and difficulty level
        4. Click 'Generate Questions' to create listening exercises
        5. Listen to the audio and select the correct answers
        6. You can also search by topic to find relevant content
        """)
    
    with st.expander("About"):
        st.markdown("""
        This app helps English speakers learn Japanese through listening comprehension exercises.
        
        Features:
        - Pull transcriptions from YouTube videos
        - Generate listening comprehension questions
        - Convert questions to audio
        - Track your progress with scoring
        - Search for specific topics in the transcript
        
        Built with ❤️ using Python, Streamlit, and OpenAI
        """)

# Footer
st.markdown("---")
st.caption("© 2023 Language Listening App | Powered by YouTube Transcript API, OpenAI, and gTTS") 