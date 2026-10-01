import pickle
import time
import os
import numpy as np
import pyautogui

from action import gojo, execute_gesture


pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

MODEL_PATH = os.path.join('models', 'gesture_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Модель распознования жестов загружена")
except FileNotFoundError:
    model = None
    print(f"Ошибка: Файл '{MODEL_PATH}' не найден! Запустите train_model.py")


COOLDOWN_TIME = 1.0 


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
    return pts.flatten().reshape(1, -1)

def process_gesture(landmarks, frame=None, timestamp_ms=0):
    """
    обработка жестов
    """

    if frame is not None and gojo.active:
        updated_frame = gojo.apply(frame, timestamp_ms)
        np.copyto(frame, updated_frame)

    if landmarks is None:
        return "Unknown", 0.0

    if model is None:
        return "Model ERROR", 0.0

    features = extract_features(landmarks)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = np.max(probabilities)

    if confidence < 0.60:
        return "Unknown", confidence
    
    execute_gesture(prediction)

    return prediction, confidence
