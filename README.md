
# Drowsiness Detection Detection (DDD)

The Drowsiness Detection Project aims to develop a real-time system for accurately detecting and alerting individuals when they exhibit signs of drowsiness. Drowsy driving is a major cause of road accidents and poses a significant threat to public safety. By leveraging computer vision and machine learning techniques, this project seeks to provide an automated solution to monitor driver's alertness levels and mitigate the risks associated with drowsiness-related incidents.

The system utilizes a camera-based approach to continuously analyze the driver's facial features, eye movements, and other behavioral cues to determine their level of drowsiness.If drowsiness is detected, an alarm sound is played to alert the driver.

By employing advanced image processing algorithms, the project identifies key facial landmarks and tracks eye movements, such as blink rate and eyelid closure duration, which are prominent indicators of drowsiness.

Additionally, other features like head pose, yawning frequency, and overall facial expressions are also considered to improve the accuracy of the detection system.


## Features
- Real-time eye aspect ratio (EAR) calculation using MediaPipe Face Mesh
- Drowsiness detection based on blink duration and EAR threshold
- Alarm sound when drowsiness is detected

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Lekhana1803/Drowsiness-Detection.git
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

## License
MIT

## References
- Dlib Face Detector
- MediaPipe Face Mesh
- OpenCV

