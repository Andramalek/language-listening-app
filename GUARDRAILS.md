# Guardrails Implementation Documentation

This document outlines the guardrails implemented in the Language Listening App to ensure security, performance, and appropriate content.

## What are Guardrails?

Guardrails are protective measures that:
- Prevent misuse of the application and its APIs
- Protect user data and system resources
- Ensure appropriate content for language learning
- Maintain application stability and performance

## Implemented Guardrails

### 1. Input Validation

**YouTube URL Validation**:
- Validates URL format before processing
- Ensures valid video ID extraction
- Prevents malformed requests

### 2. API Rate Limiting

**OpenAI API Protection**:
- Limits requests per minute: `MAX_API_REQUESTS_PER_MINUTE=20`
- Limits requests per day: `MAX_API_REQUESTS_PER_DAY=1000`
- Provides clear error messages when limits are reached

### 3. Content Filtering

**Inappropriate Content Detection**:
- Scans transcripts for inappropriate keywords
- Filters out videos with excessive profanity
- Can be enabled/disabled with `ENABLE_CONTENT_FILTER=true|false`

### 4. Resource Management

**Application Usage Limits**:
- Maximum questions per session: `MAX_QUESTIONS_PER_SESSION=20`
- Maximum audio files per session: `MAX_AUDIO_FILES_PER_SESSION=50`
- Maximum search queries per minute: `MAX_SEARCH_QUERIES_PER_MINUTE=10`

### 5. Error Handling

**Robust Error Tracking**:
- Logs all errors with appropriate context
- Provides user-friendly error messages
- Prevents application crashes
- Monitors error frequency to detect potential issues

## Configuration

To configure guardrail settings, add the following to your `.env` file:

```
# API usage limits
MAX_API_REQUESTS_PER_MINUTE=20
MAX_API_REQUESTS_PER_DAY=1000

# Content filtering
ENABLE_CONTENT_FILTER=true

# Application limits
MAX_QUESTIONS_PER_SESSION=20
MAX_AUDIO_FILES_PER_SESSION=50
MAX_SEARCH_QUERIES_PER_MINUTE=10
```

## Benefits

- **Security**: Protects the application from potential misuse
- **Cost Control**: Prevents excessive API usage
- **Stability**: Ensures consistent performance under varying loads
- **User Experience**: Provides clear feedback when limits are reached
- **Appropriateness**: Ensures content is suitable for educational purposes 