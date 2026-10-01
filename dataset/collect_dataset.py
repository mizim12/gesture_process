import csv
import cv2
import mediapipe as mp
import time
import numpy as np
import os


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join('models', 'hand_landmarker.task')

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

GESTURE_NAMES = [
    "Кулак",
    "Лайк",
    "Дизлайк",
    "Расширение территории",
    "OK",
    "Ладонь",
    "Мир",
    "Указательный вверх",
    "Коза",
    "Звонок"
]

KEY_MAPPING = {ord(str(i)): name for i, name in enumerate(GESTURE_NAMES)}

CSV_FILE = 'dataset/gestures_dataset.csv'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def extract_features(landmarks):
    """
    Инвариантность к сдвигу, масштабу и повороту руки.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    pts = pts - pts[0]

    v_dir = pts[9]
    scale = np.linalg.norm(v_dir[:2])
    if scale > 0:
        pts = pts / scale

    angle = np.arctan2(v_dir[0], -v_dir[1])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

    pts[:, :2] = np.dot(pts[:, :2], rotation_matrix.T)
    return pts.flatten().tolist()


def count_recorded_samples():
    """
    Считаем число уже полученных кадров для класса
    """
    counts = {name: 0 for name in GESTURE_NAMES}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts

cap = cv2.VideoCapture(0)
recorded_counts = count_recorded_samples()


with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_landmarks = None

        if results.hand_landmarks:
            current_landmarks = results.hand_landmarks[0]

            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = (int(current_landmarks[start_idx].x * w), int(current_landmarks[start_idx].y * h))
                pt2 = (int(current_landmarks[end_idx].x * w), int(current_landmarks[end_idx].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            for lm in current_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        active_gesture = None
        if key in KEY_MAPPING and current_landmarks:
            active_gesture = KEY_MAPPING[key]

            row = [active_gesture] + extract_features(current_landmarks)
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            recorded_counts[active_gesture] += 1


        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 310), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Шпаргалка клавиш:", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset = 65
        for idx, name in enumerate(GESTURE_NAMES):
            key_num = idx
            count = recorded_counts[name]

            color = (0, 255, 0) if name == active_gesture else (220, 220, 220)
            text = f"[{key_num}] {name}: {count} кадров"

            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 26

        if active_gesture:
            cv2.putText(frame, f"ЗАПИСЬ: {active_gesture}", (340, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Dataset Collector", frame)

cap.release()
cv2.destroyAllWindows()
