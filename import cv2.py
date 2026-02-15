
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import ctypes
import math
import os
import time
import urllib.request
from pynput.mouse import Button, Controller


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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera device 0")

pTime = 0
cTime = 0

# cam and mouse settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 780
SMOOTHING = 0.25
CLICK_COOLDOWN = 0.3
USE_RAW_INPUT = True
MOUSE_SENSITIVITY = 1.0
mouse = Controller()
prev_mouse_x = SCREEN_WIDTH // 2
prev_mouse_y = SCREEN_HEIGHT // 2
prev_index_bent = False
prev_middle_pinch = False
last_left_click = 0.0
last_right_click = 0.0

if USE_RAW_INPUT:
    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

    def send_mouse_move(dx, dy):
        if dx == 0 and dy == 0:
            return
        inp = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=int(dx),
                dy=int(dy),
                mouseData=0,
                dwFlags=MOUSEEVENTF_MOVE,
                time=0,
                dwExtraInfo=None
            )
        )
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def send_mouse_click(is_left):
        if is_left:
            down_flag = MOUSEEVENTF_LEFTDOWN
            up_flag = MOUSEEVENTF_LEFTUP
        else:
            down_flag = MOUSEEVENTF_RIGHTDOWN
            up_flag = MOUSEEVENTF_RIGHTUP

        inp_down = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=0,
                dy=0,
                mouseData=0,
                dwFlags=down_flag,
                time=0,
                dwExtraInfo=None
            )
        )
        inp_up = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=0,
                dy=0,
                mouseData=0,
                dwFlags=up_flag,
                time=0,
                dwExtraInfo=None
            )
        )
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))

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
        hand_landmarks = results.hand_landmarks[0]

        landmark_points = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_points.append((cx, cy))
            cv2.circle(img, (cx, cy), 5, (255, 188, 217), cv2.FILLED)

        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                cv2.line(img, start_point, end_point, (0, 0, 0), 2)

        palm_ids = [0, 5, 9, 13, 17]
        palm_x = sum(hand_landmarks[i].x for i in palm_ids) / len(palm_ids)
        palm_y = sum(hand_landmarks[i].y for i in palm_ids) / len(palm_ids)
        target_x = int((1.0 - palm_x) * SCREEN_WIDTH)
        target_y = int(palm_y * SCREEN_HEIGHT)
        new_mouse_x = int(prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING)
        new_mouse_y = int(prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING)
        if USE_RAW_INPUT:
            dx = (new_mouse_x - prev_mouse_x) * MOUSE_SENSITIVITY
            dy = (new_mouse_y - prev_mouse_y) * MOUSE_SENSITIVITY
            send_mouse_move(dx, dy)
        else:
            mouse.position = (new_mouse_x, new_mouse_y)
        prev_mouse_x = new_mouse_x
        prev_mouse_y = new_mouse_y

        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        index_pip = hand_landmarks[6]
        index_mcp = hand_landmarks[5]
        middle_tip = hand_landmarks[12]

        index_bent = (index_tip.y - index_pip.y) > 0.03 and (index_tip.y - index_mcp.y) > 0.05
        middle_pinch = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y) < 0.04

        now = time.time()
        if index_bent and not prev_index_bent and now - last_left_click > CLICK_COOLDOWN:
            if USE_RAW_INPUT:
                send_mouse_click(True)
            else:
                mouse.click(Button.left, 1)
            last_left_click = now

        if middle_pinch and not prev_middle_pinch and now - last_right_click > CLICK_COOLDOWN:
            if USE_RAW_INPUT:
                send_mouse_click(False)
            else:
                mouse.click(Button.right, 1)
            last_right_click = now

        prev_index_bent = index_bent
        prev_middle_pinch = middle_pinch
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
