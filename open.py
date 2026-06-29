import cv2
import os
import time
from datetime import datetime

# ===========================
# Create folders if missing
# ===========================

os.makedirs("saved_faces", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ===========================
# Load Face Detector
# ===========================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ===========================
# Start Webcam
# ===========================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("Cannot open camera")
    exit()

# ===========================
# Variables
# ===========================

font = cv2.FONT_HERSHEY_SIMPLEX

saved_count = 0

prev_time = time.time()

# ===========================
# Main Loop
# ===========================

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=6,
        minSize=(60, 60)
    )

    # FPS

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # Draw Face Boxes

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face = frame[y:y+h, x:x+w]

        filename = f"saved_faces/face_{saved_count}.jpg"

        cv2.imwrite(filename, face)

        saved_count += 1

        cv2.putText(
            frame,
            "FACE",
            (x, y - 10),
            font,
            0.7,
            (0, 255, 0),
            2
        )

    # Timestamp

    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    cv2.putText(
        frame,
        timestamp,
        (20, 35),
        font,
        0.7,
        (255, 255, 255),
        2
    )

    # Face Count

    cv2.putText(
        frame,
        f"Faces : {len(faces)}",
        (20, 70),
        font,
        0.8,
        (0, 255, 255),
        2
    )

    # FPS

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 105),
        font,
        0.8,
        (255, 255, 0),
        2
    )

    # Instructions

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, 140),
        font,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press S to Save Screenshot",
        (20, 175),
        font,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Advanced Face Detection System", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    elif key == ord('s'):

        name = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(
            f"screenshots/{name}.jpg",
            frame
        )

        print("Screenshot Saved")

# ===========================
# Cleanup
# ===========================

camera.release()

cv2.destroyAllWindows()