# ASL Finger Spelling Agent

## Accomplishments

I've built the ASL Finger Spelling Agent Application using Streamlit, allowing users to practice ASL fingerspelling with real-time feedback. Key achievements include:

- **Functional App Structure** – Two practice modes: guided (learning specific signs) and free practice.
- **Hand Tracking** – Integrated MediaPipe to detect 21 hand landmarks in real-time.
- **Sign Recognition Model** – Trained on a public ASL dataset, achieving 85% accuracy for static signs.
- **Threading Optimization** – Separated webcam capture, hand detection, and UI rendering for better performance.
- **WSL Webcam Bridge** – Developed a workaround to enable webcam access in WSL.
- **AI-Assisted Development** – Used Cursor AI to generate code templates, debug, and optimize the MediaPipe integration.

## Challenges & Uncertainties

Despite progress, several technical hurdles remain:

- **Model Accuracy Issues** – Struggles with similar hand shapes (e.g., L/V/E) despite data augmentation attempts.
- **Dynamic Signs** – Letters J & Z require motion tracking, which I haven't fully implemented.
- **Camera Integration** – WSL webcam bridging was complex; unsure if my approach is optimal.
- **Session Management** – Streamlit's session state handling was confusing, leading to occasional resets.
- **Dataset Handling** – Large ASL datasets (>2GB) caused issues with GitHub, exposing my need for better GitHub proficiency.
- **AI Code Assistance Limitations** – Cursor AI was helpful but required a deeper understanding of the underlying concepts.
- **Model Testing** – Accuracy for some letters is low:
  - N (44%) – Confused with M
  - U (27%) – Confused with V
  - M (55%) – Confused with N

## Final Thoughts

I'm proud of the progress but recognize major areas for improvement, especially in computer vision, model training, and app architecture. While AI tools accelerate development, a strong programming foundation remains essential.

## Setup & Usage

### Prerequisites
- Python 3.8+
- Webcam

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/asl-finger-spelling-agent.git

# Navigate to the project directory
cd asl-finger-spelling-agent

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

### WSL Users
If running in WSL, you'll need to use the Windows Camera Bridge:

1. Copy `windows_camera_bridge.py` to your Windows desktop
2. Run `debug_camera_bridge.bat` on Windows
3. Set the camera URL environment variable:
   ```bash
   CAMERA_URL="http://YOUR_WINDOWS_IP:8080/video_feed" streamlit run app.py
   ```

## Future Improvements

- Improve accuracy for confusable letters
- Add support for dynamic signs (J, Z)
- Implement motion tracking for full fingerspelling sentences
- Add user accounts to track progress
- Expand to include common ASL words and phrases

## Features

- Real-time hand detection and tracking using MediaPipe
- ASL fingerspelling recognition using machine learning
- Two practice modes: Guided and Free Practice
- Instant feedback on your signing accuracy
- User-friendly interface with Streamlit

## Getting Started

### Prerequisites

- Python 3.8+ installed
- Webcam connected to your computer
- Git (optional, for cloning the repository)

### Installation

1. Clone the repository or download the source code:
   ```bash
   git clone https://github.com/yourusername/asl-finger-spelling-agent.git
   cd asl-finger-spelling-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Run the application using Streamlit:

```bash
cd finger_spelling_agent
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. **Choose a Practice Mode**:
   - **Guided Practice**: The app will show you a letter to sign
   - **Free Practice**: Practice any signs you want with real-time recognition

2. **Start the Application**:
   - Click the "Start" button in the sidebar
   - Allow webcam access if prompted

3. **Practice ASL Fingerspelling**:
   - Position your hand in the camera view
   - Make ASL fingerspelling signs with your hand
   - Get real-time feedback on your signing

4. **Adjust Settings**:
   - Change detection confidence for better recognition
   - Select between different recognition models

## Technical Details

### Components

- **WebcamCapture**: Handles capturing video from the webcam
- **HandDetector**: Detects and extracts hand landmarks using MediaPipe
- **SignRecognizer**: Recognizes ASL fingerspelling signs using ML models
- **Visualization**: Provides visual feedback on the detected signs

### Recognition Methods

The application supports two recognition methods:

1. **MediaPipe Landmarks**: Uses the 3D hand landmarks from MediaPipe for recognition
2. **CNN Classifier**: Uses a convolutional neural network for image-based recognition

## Development Notes

### Training Custom Models

To train custom models with your own data:

1. Collect ASL fingerspelling images for each letter
2. For landmark-based model, extract landmarks using MediaPipe
3. Train using the provided utility functions in `sign_recognizer.py`

### Future Improvements

- Support for full ASL words and phrases
- Improved recognition for letters that involve motion (J, Z)
- User accounts to save progress
- Customizable practice sessions

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [American Sign Language University](https://www.lifeprint.com/) 