
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import urllib.request


# hand finger/palm connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17)  # palm
]

# setup hand model
model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"hand_landmarker.task not found at {model_path}")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(options)

# setup face model
face_model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
if not os.path.isfile(face_model_path):
    face_model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(face_model_url, face_model_path)
face_base_options = python.BaseOptions(model_asset_path=face_model_path)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera device 0")

pTime = 0
cTime = 0

# main loop

while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    results = hand_detector.detect(mp_image)
    face_results = face_detector.detect(mp_image)

    if results.hand_landmarks:
        h, w, c = img.shape
# draw hand landmarks and finger conneciton
        for hand_landmarks in results.hand_landmarks:
            landmark_points = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_points.append((cx, cy))
                cv2.circle(img, (cx, cy), 5, (255,188,217), cv2.FILLED)

            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    start_point = landmark_points[start_idx]
                    end_point = landmark_points[end_idx]
                    cv2.line(img, start_point, end_point, (0, 0, 0), 2)
# draw face landmarks
    if face_results.face_landmarks:
        h, w, c = img.shape
        for face_landmarks in face_results.face_landmarks:
            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 1, (255, 255, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
