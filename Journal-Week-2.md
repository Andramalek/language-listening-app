Hypothesis and Technical Uncertainty
Starting this project, I had several questions:
Could I build a system that processes real Japanese audio and text?
Would it be possible to analyze handwritten Japanese characters?
Could I create a real-time hand tracking system for ASL?
My main hypotheses were:
YouTube transcripts could be used for listening practice
Computer vision could analyze Japanese handwriting
MediaPipe could track hand positions for ASL
Technical Exploration
Language Listening App:
Used YouTube API to get video transcripts
Implemented vector database for storing transcripts
Created text-to-speech conversion
Built a Streamlit interface
Integrated OpenAI for question generation
Writing Practice App:
Built sentence generation system
Implemented two OCR options:
MangaOCR for local processing
OpenAI Vision for cloud processing
Created image upload functionality
Made a feedback system for writing accuracy
Finger Spelling Agent:
Integrated MediaPipe for hand tracking
Built two practice modes:
Guided learning
Free practice
Implemented real-time sign recognition
Created WSL webcam bridge
Used threading for better performance
Final Observations and Outcomes
What Worked Well:
Language Listening App:
Successfully pulls YouTube transcripts
Generates good practice questions
Interface is easy to use
Text-to-speech works well
Writing Practice App:
OCR accurately reads Japanese writing
Sentence generation is helpful
Feedback system works as intended
Multiple OCR options give flexibility
Finger Spelling Agent:
Achieves 85% accuracy for static signs
Real-time hand tracking works
Threading improved performance
WSL webcam bridge functions
Areas for Improvement:
Language Listening App:
Could add more practice formats
Need better transcript filtering
Audio sync needs work
Database optimization needed
Writing Practice App:
OCR sometimes misreads complex kanji
Sentence generation could be more varied
Need better error handling
Could add stroke order guidance
Finger Spelling Agent:
Some letters have low accuracy:
N (44%) - Gets mixed up with M
U (27%) - Gets mixed up with V
M (55%) - Gets mixed up with N
Dynamic signs (J & Z) need work
Session management is inconsistent
Future Plans:
Improve model accuracy
Add more practice features
Optimize performance
Create better documentation
I learned a lot about:
Working with APIs
Computer vision
Machine learning
User interface design
Threading and performance
Error handling
