# Writing Practice App

A language learning application for practicing writing Japanese sentences. This app complements the Language Listening App by focusing on writing skills.

## Features

- Generate simple English sentences based on word groups
- Translate sentences to Japanese for writing practice
- Upload images of handwritten Japanese text
- Analyze handwriting using OCR (MangaOCR or OpenAI Vision)
- Get feedback on writing accuracy

## How It Works

1. Enter words in the sidebar and generate sentences
2. Practice writing the Japanese translations on paper
3. Take a photo or scan of your handwriting
4. Upload the image for each sentence
5. Get instant feedback on your handwriting

## Technical Details

### Components

- **Streamlit**: Frontend interface
- **SentenceGenerator**: Generates simple sentences and translations
- **OCRProcessor**: Analyzes handwritten Japanese text
- **MangaOCR**: Japanese OCR engine (optional)
- **OpenAI Vision**: Alternative OCR using GPT-4V

### OCR Options

The app supports two OCR engines:

1. **MangaOCR**: Specialized for Japanese text recognition
   - Better for handwritten Japanese
   - Runs locally (requires installation)
   - No API costs

2. **OpenAI Vision (GPT-4V)**: 
   - More flexible but less specialized
   - Requires OpenAI API key
   - Incurs API usage costs

## Installation

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/language-learning-app.git
cd language-learning-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### MangaOCR Setup (Optional but Recommended)

To use MangaOCR for better Japanese handwriting recognition:

1. Uncomment the manga-ocr line in requirements.txt
2. Install with PyTorch dependencies:

```bash
pip install manga-ocr
```

Note: MangaOCR requires PyTorch, which may take some time to install.

## Usage

```bash
# Run the Writing Practice app
streamlit run writing_practice.py
```

## Configuration

Create a `.env` file with the following settings:

```
# OpenAI API key (required for translations and OpenAI Vision OCR)
OPENAI_API_KEY=your_openai_api_key

# OCR engine selection (true for MangaOCR, false for OpenAI Vision)
USE_MANGAOCR=true
```

## Tips for Better Recognition

- Write clearly and avoid connecting characters
- Make sure your paper is well-lit and the writing is visible
- Take photos straight-on, not at an angle
- Use a dark pen or marker for better contrast
- Practice with simple sentences first 