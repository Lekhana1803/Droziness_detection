
# Fatigue (Drowsiness) Detection System

This project uses computer vision to detect driver drowsiness in real-time using a webcam. If drowsiness is detected, an alarm sound is played to alert the driver.

## Features
- Real-time eye aspect ratio (EAR) calculation using MediaPipe Face Mesh
- Drowsiness detection based on blink duration and EAR threshold
- Alarm sound when drowsiness is detected

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/GouthamShanubhogar/Drowsiness-Detection.git
cd Fatigue-Detection-System-Based-On-Behavioural-Characteristics-Of-Driver-master
```

### 2. Create and Activate Python Virtual Environment
```bash
python -m venv venv310
source venv310/Scripts/activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install opencv-python mediapipe numpy scipy playsound
```

### 4. Download Model Files
- Place the `shape_predictor_70_face_landmarks.dat` file in the `models/` directory (already included).
- Ensure `alarm.wav` is present in the project root for the alarm sound.

## How to Run
```bash
python blinkDetect.py
```

## Notes
- Requires a working webcam.
- Tested on Python 3.10+.
- For best results, run in a well-lit environment.

## Troubleshooting
- If you get errors about missing packages, ensure you are in the correct virtual environment and have installed all dependencies.
- If the alarm sound does not play, check that `alarm.wav` exists and is a valid audio file.

## License
MIT

## References
- Dlib Face Detector
- MediaPipe Face Mesh
- OpenCV

---
For more details, see the code and comments in `blinkDetect.py`.
