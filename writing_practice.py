"""
Writing Practice App - Main Streamlit Application

A language learning application for practicing writing Japanese sentences.
Users can input word groups, get simple English sentences, and upload images of their handwritten Japanese.
"""

import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv
import io

# Import utility modules
from writing_utils import SentenceGenerator, OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Writing Practice App",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize app components
@st.cache_resource
def load_sentence_generator():
    return SentenceGenerator()

@st.cache_resource
def load_ocr_processor():
    # Determine which OCR engine to use
    use_mangaocr = os.getenv("USE_MANGAOCR", "true").lower() == "true"
    return OCRProcessor(use_mangaocr=use_mangaocr)

# Get instances of our utility classes
sentence_generator = load_sentence_generator()
ocr_processor = load_ocr_processor()

# Initialize session state
if 'word_groups' not in st.session_state:
    st.session_state.word_groups = []
if 'english_sentences' not in st.session_state:
    st.session_state.english_sentences = []
if 'japanese_translations' not in st.session_state:
    st.session_state.japanese_translations = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = {}
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# Main application
st.title("✍️ Writing Practice App")
st.subheader("Practice writing Japanese sentences")

# Sidebar for inputs
with st.sidebar:
    st.header("Word Groups")
    st.markdown("""
    Enter words to generate simple sentences for writing practice.
    Words can be in English or Japanese.
    """)
    
    word_input = st.text_input("Enter words separated by commas:", placeholder="cat, dog, house, book")
    
    sentences_per_word = st.slider("Sentences per word:", 1, 3, 1)
    
    if st.button("Generate Sentences"):
        if word_input:
            # Split the input into a list of words
            words = [word.strip() for word in word_input.split(",") if word.strip()]
            
            if words:
                with st.spinner("Generating sentences and translations..."):
                    st.session_state.word_groups = words
                    
                    # Generate English sentences
                    english_sentences = sentence_generator.generate_sentences(
                        words, 
                        count_per_word=sentences_per_word
                    )
                    
                    # Translate to Japanese
                    japanese_translations = sentence_generator.translate_to_japanese(
                        english_sentences
                    )
                    
                    # Store in session state
                    st.session_state.english_sentences = english_sentences
                    st.session_state.japanese_translations = japanese_translations
                    
                    # Clear previous results
                    st.session_state.uploaded_images = {}
                    st.session_state.ocr_results = {}
                    st.session_state.feedback = {}
                    
                    st.success(f"Generated {len(english_sentences)} sentences for practice.")
            else:
                st.error("Please enter valid words separated by commas.")
        else:
            st.error("Please enter some words to generate sentences.")
    
    # OCR Engine selection
    st.header("OCR Settings")
    ocr_choice = st.radio(
        "Choose OCR Engine:",
        ["MangaOCR (Recommended for Japanese)", "OpenAI Vision (GPT-4V)"]
    )
    
    if ocr_choice == "MangaOCR (Recommended for Japanese)":
        if st.button("Use MangaOCR"):
            ocr_processor.use_mangaocr = True
            try:
                # Check if MangaOCR is available
                from manga_ocr import MangaOcr
                st.success("MangaOCR selected.")
            except ImportError:
                st.error("MangaOCR not installed. Using OpenAI Vision as fallback.")
                ocr_processor.use_mangaocr = False
    else:
        if st.button("Use OpenAI Vision"):
            ocr_processor.use_mangaocr = False
            st.success("OpenAI Vision selected.")

# Main content area
tab1, tab2 = st.tabs(["Practice", "Results"])

with tab1:
    st.header("Writing Practice")
    
    if st.session_state.english_sentences:
        for i, (eng, jpn) in enumerate(zip(st.session_state.english_sentences, 
                                         st.session_state.japanese_translations)):
            with st.expander(f"Sentence {i+1}", expanded=(i==0)):
                st.markdown(f"**English:** {eng}")
                
                # Replace nested expander with checkbox for translation hint
                show_hint = st.checkbox(f"Show Japanese translation (hint)", key=f"hint_{i}")
                if show_hint:
                    st.markdown(f"**Japanese:** {jpn}")
                
                # Image upload for this sentence
                st.markdown("### Upload your handwritten Japanese")
                st.markdown("Write the Japanese sentence on paper and upload a photo.")
                
                uploaded_file = st.file_uploader(f"Upload image for sentence {i+1}", 
                                               type=["jpg", "jpeg", "png"],
                                               key=f"uploader_{i}")
                
                if uploaded_file:
                    # Display the uploaded image
                    st.image(uploaded_file, caption=f"Your writing for sentence {i+1}", width=400)
                    
                    # Store the uploaded image in session state
                    # Convert uploaded file to bytes
                    bytes_data = uploaded_file.getvalue()
                    st.session_state.uploaded_images[i] = bytes_data
                    
                    # Button to analyze the handwriting
                    if st.button("Analyze Handwriting", key=f"analyze_{i}"):
                        with st.spinner("Analyzing your handwriting..."):
                            try:
                                # Process the image with OCR
                                image_data = io.BytesIO(bytes_data)
                                
                                # Log image details for debugging
                                logger.info(f"Processing image for sentence {i+1}")
                                logger.info(f"Image size: {len(bytes_data)} bytes")
                                
                                # Reset the buffer position to the beginning
                                image_data.seek(0)
                                
                                expected_text = st.session_state.japanese_translations[i]
                                logger.info(f"Expected text: {expected_text}")
                                
                                # Call OCR processor
                                ocr_result = ocr_processor.process_image(
                                    image_data,
                                    expected_text=expected_text
                                )
                                
                                # Store the result
                                st.session_state.ocr_results[i] = ocr_result
                                
                                # Display the result
                                if 'text' in ocr_result:
                                    st.markdown(f"**Detected text:** {ocr_result['text']}")
                                else:
                                    st.error("No text was detected in the image.")
                                
                                if 'feedback' in ocr_result:
                                    st.session_state.feedback[i] = ocr_result['feedback']
                                    st.success(ocr_result['feedback'])
                                
                                if 'error' in ocr_result and ocr_result['error']:
                                    error_msg = ocr_result.get('feedback', 'Unknown error')
                                    st.error(f"Error during OCR processing: {error_msg}")
                                    logger.error(f"OCR processing error: {error_msg}")
                                
                            except Exception as e:
                                logger.error(f"Error analyzing handwriting: {e}", exc_info=True)
                                st.error(f"Error analyzing handwriting: {str(e)}")
                                st.info("Try with a different image or check your OpenAI API key.")
    else:
        st.info("Enter words in the sidebar and click 'Generate Sentences' to start practicing.")

with tab2:
    st.header("Your Progress")
    
    if st.session_state.ocr_results:
        st.markdown("### Writing Analysis")
        
        # Show a summary table
        st.markdown("#### Summary")
        summary_data = []
        
        for i, ocr_result in st.session_state.ocr_results.items():
            if i < len(st.session_state.english_sentences):
                english = st.session_state.english_sentences[i]
                expected = st.session_state.japanese_translations[i]
                detected = ocr_result.get('text', 'N/A')
                feedback = st.session_state.feedback.get(i, 'No feedback')
                
                # Calculate a simple accuracy score
                detected_clean = ''.join(detected.split())
                expected_clean = ''.join(expected.split())
                correct_chars = sum(1 for a, b in zip(detected_clean, expected_clean) if a == b)
                max_length = max(len(detected_clean), len(expected_clean))
                accuracy = correct_chars / max_length if max_length > 0 else 0
                
                summary_data.append([
                    f"Sentence {i+1}",
                    english,
                    expected,
                    detected,
                    f"{accuracy:.0%}"
                ])
        
        if summary_data:
            st.table({
                "Sentence": [row[0] for row in summary_data],
                "English": [row[1] for row in summary_data],
                "Expected Japanese": [row[2] for row in summary_data],
                "Your Writing": [row[3] for row in summary_data],
                "Accuracy": [row[4] for row in summary_data]
            })
        
        # Detailed results
        st.markdown("#### Detailed Results")
        for i, ocr_result in st.session_state.ocr_results.items():
            with st.expander(f"Sentence {i+1} Details"):
                if i < len(st.session_state.english_sentences):
                    st.markdown(f"**English:** {st.session_state.english_sentences[i]}")
                    st.markdown(f"**Expected Japanese:** {st.session_state.japanese_translations[i]}")
                    st.markdown(f"**Your writing (OCR result):** {ocr_result.get('text', 'N/A')}")
                    
                    if 'confidence' in ocr_result:
                        st.progress(float(ocr_result['confidence']))
                        st.text(f"Confidence score: {ocr_result['confidence']:.2f}")
                    
                    if i in st.session_state.feedback:
                        st.markdown(f"**Feedback:** {st.session_state.feedback[i]}")
                    
                    if i in st.session_state.uploaded_images:
                        st.image(st.session_state.uploaded_images[i], caption="Your handwriting", width=300)
    else:
        st.info("After analyzing your handwriting, results will appear here.")

st.markdown("---")
st.markdown("### How to use this app")
st.markdown("""
1. Enter words in the sidebar and click 'Generate Sentences'
2. Practice writing the Japanese translations on paper
3. Take a photo or scan of your handwriting
4. Upload the image for each sentence
5. Click 'Analyze Handwriting' to get feedback
""")

st.markdown("---")
st.markdown("### Tips for better recognition")
st.markdown("""
- Write clearly and avoid connecting characters
- Make sure your paper is well-lit and the writing is visible
- Take photos straight-on, not at an angle
- Use a dark pen or marker for better contrast
- Practice with simple sentences first
""") 