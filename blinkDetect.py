import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import time
import queue
import winsound  # <-- Added import

# --- Configuration ---
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
ALARM_SOUND_PATH = "alarm.wav"

# MediaPipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eyes landmarks from MediaPipe (right and left)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Thread-safe alarm system

alarm_on = False
alarm_thread = None


def sound_alert(path):
    # Play sound in a loop until stopped
    while alarm_on:
        winsound.PlaySound(path, winsound.SND_FILENAME)

def alarm_thread_func(path, thread_status_q):
    while True:
        if not thread_status_q.empty():
            finished = thread_status_q.get()
            if finished:
                break
        sound_alert(path)

def eye_aspect_ratio(eye):
    # compute euclidean distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def extract_eye_coords(landmarks, eye_indices, shape):
    h, w = shape
    coords = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        coords.append((x, y))
    return coords

# Main camera feed
cap = cv2.VideoCapture(0)
COUNTER = 0
DROWSY = False

print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            shape = frame.shape[:2]
            left_eye = extract_eye_coords(face_landmarks.landmark, LEFT_EYE, shape)
            right_eye = extract_eye_coords(face_landmarks.landmark, RIGHT_EYE, shape)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not DROWSY:
                        DROWSY = True
                        if not alarm_on:
                            alarm_on = True
                            alarm_thread = Thread(target=sound_alert, args=(ALARM_SOUND_PATH,))
                            alarm_thread.daemon = True
                            alarm_thread.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                COUNTER = 0
                DROWSY = False
                if alarm_on:
                    alarm_on = False
                    # Wait for thread to finish
                    if alarm_thread is not None:
                        alarm_thread.join()

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        cv2.putText(frame, "No face detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        thread_status_q.put(True)
        break

cap.release()
cv2.destroyAllWindows()
