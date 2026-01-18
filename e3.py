import cv2
from deepface import DeepFace
import numpy as np
import time

# Face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

frame_count = 0
last_emotions = {}           # Stores latest emotion for each face (no blinking)
stress_levels = {}           # Stores stress for each face
emotion_history = {}         # For mental health alerts

def get_stress_from_emotion(em):
    if em in ["angry", "fear", "sad"]:
        return 2
    elif em in ["neutral"]:
        return 1
    else:
        return 0

def get_redness_level(face_roi):
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    # Red color ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_pixels = cv2.countNonZero(mask1 + mask2)

    redness_score = red_pixels / (face_roi.shape[0]*face_roi.shape[1] + 1)

    return round(redness_score * 100, 1)   # like heart-rate intensity

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    frame_count += 1

    face_id = 0

    for (x, y, w, h) in faces:
        face_id += 1
        face_label = f"Face {face_id}"
        
        face_roi = frame[y:y+h, x:x+w]

        # Analyze every 15 frames (reduces blinking)
        if frame_count % 15 == 0:
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                if isinstance(result, list):
                    emotion = result[0]['dominant_emotion']
                else:
                    emotion = result['dominant_emotion']

                last_emotions[face_label] = emotion

                # A. Stress level from emotion
                stress_level = get_stress_from_emotion(emotion)
                stress_levels[face_label] = stress_level

                # B. Mental health: track sadness, fear, anger
                if face_label not in emotion_history:
                    emotion_history[face_label] = []

                emotion_history[face_label].append(emotion)
                if len(emotion_history[face_label]) > 30:
                    emotion_history[face_label].pop(0)

                # C. Redness-based heart-rate-like stress
                redness = get_redness_level(face_roi)
                stress_levels[face_label] += redness / 50  # combine both

            except:
                pass

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display stored (non-blinking) emotion
        emotion_text = last_emotions.get(face_label, "Detecting...")

        # Stress score
        stress_text = f"Stress: {round(stress_levels.get(face_label, 0), 2)}"

        # Mental health warning
        warning = ""
        if face_label in emotion_history:
            if emotion_history[face_label].count("sad") > 15:
                warning = "Warning: Persistent Sadness"
            elif emotion_history[face_label].count("fear") > 15:
                warning = "Warning: Fear Detected"
            elif emotion_history[face_label].count("angry") > 15:
                warning = "Warning: Anger Detected"

        # Show emotion
        cv2.putText(frame, f"{face_label}: {emotion_text}", (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Show stress
        cv2.putText(frame, stress_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        # Show mental health alert
        if warning != "":
            cv2.putText(frame, warning, (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Advanced Emotion Health & Stress Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
