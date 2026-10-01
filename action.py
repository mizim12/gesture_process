import pyautogui
import ctypes
import os
import time
import threading
from collections import deque, Counter
from domain_expansion import Gojo_Effect

pyautogui.FAILSAFE = False

gojo = Gojo_Effect()

actions_enabled = False

last_execution_times = {}

WINDOW_SIZE = 15
THRESHOLD = 10
gesture_history = deque(maxlen=WINDOW_SIZE)

def is_enabled(func):
    """
    Декоратор режима действия
    """
    def wrapper(*args, **kwargs):
        if not actions_enabled:
            return None
        return func(*args, **kwargs)
    return wrapper

def action_peace():
    global actions_enabled
    actions_enabled = not actions_enabled

@is_enabled
def action_fist():
    pyautogui.press('playpause')

@is_enabled
def action_open():
    pyautogui.press('volumemute')

@is_enabled
def action_like():
    pyautogui.press('volumeup')

@is_enabled
def action_dislike():
    pyautogui.press('volumedown')

@is_enabled
def action_domain():
    gojo.trigger()

@is_enabled
def action_ok():
    pyautogui.press('l')

@is_enabled
def action_goat():
    pyautogui.press('f')

@is_enabled
def action_index():
    folder = "screenshots"
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"screenshot_{int(time.time())}.png")
    pyautogui.screenshot(filename)
    print(f"Скриншот сохранён: {filename}")


@is_enabled
def action_ring():
    def _open_ring_file_delayed():
        time.sleep(2.0)
        video_path = os.path.join('assets', 'ring.png')
        if os.path.exists(video_path):
            os.startfile(video_path)
        else:
            print(f"Ошибка: Файл '{video_path}' не найден в папке проекта!")
    threading.Thread(target=_open_ring_file_delayed, daemon=True).start()

ACTIONS = {
    "Кулак":                 {"action": action_fist,    "cooldown": 3},
    "Ладонь":                {"action": action_open,    "cooldown": 3},
    "Лайк":                  {"action": action_like,    "cooldown": 0.15},
    "Дизлайк":               {"action": action_dislike, "cooldown": 0.15},
    "Расширение территории": {"action": action_domain,  "cooldown": 8},
    "OK":                    {"action": action_ok,      "cooldown": 1},
    "Мир":                   {"action": action_peace,   "cooldown": 5},
    "Указательный вверх":    {"action": action_index,   "cooldown": 2},
    "Коза":                  {"action": action_goat,    "cooldown": 3},
    "Звонок":                {"action": action_ring,    "cooldown": 10},
}

def execute_gesture(gesture_name):
    if gesture_name not in ACTIONS:
        return

    gesture_history.append(gesture_name)

    counts = Counter(gesture_history)
    most_common_gesture, count = counts.most_common(1)[0]
    
    if count >= THRESHOLD:
        gesture_config = ACTIONS[most_common_gesture]
        
        gesture_config = ACTIONS[gesture_name]
        action_func = gesture_config["action"]
        cooldown = gesture_config["cooldown"]

        current_time = time.time()
        last_time = last_execution_times.get(gesture_name, 0.0)

        if (current_time - last_time) >= cooldown:
            last_execution_times[gesture_name] = current_time
            action_func()
