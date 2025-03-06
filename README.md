# Language Listening App

A language learning application for English speakers learning Japanese, using YouTube video transcripts to generate listening comprehension exercises.

## Features

- Pull transcriptions from YouTube videos
- Store transcripts in a vector database for semantic search
- Generate listening comprehension questions based on the transcripts
- Convert questions to audio using text-to-speech
- User-friendly interface built with Streamlit

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the application:
   ```
   streamlit run main.py
   ```

## Project Structure

- `main.py`: Streamlit application entry point
- `utils/`: Utility functions for transcript fetching, embedding generation, etc.
- `data/`: Directory for storing the vector database

## Requirements

- Python 3.8+
- YouTube Transcript API
- OpenAI API key for embeddings and question generation
- Internet connection for YouTube transcript fetching and TTS 