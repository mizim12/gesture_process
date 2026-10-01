import cv2
import numpy as np
import time
import mediapipe as mp
import threading
import os
import winsound

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join('models', 'face_landmarker.task')
class Gojo_Effect:
    def __init__(self, duration=10, face_model_path=MODEL_PATH, sound_path="gojo_voice.wav"):
        self.duration = duration
        self.active = False
        self.sound_path = sound_path
        self.end_time = 0.0

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )
        
        self.landmarker = FaceLandmarker.create_from_options(options)

        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]

        self.LEFT_EYE_FULL = [33, 133, 159, 145, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155]
        self.RIGHT_EYE_FULL = [362, 263, 386, 374, 387, 388, 466, 249, 390, 373, 374, 380, 381, 382]

    def _play_sound(self):
        if os.path.exists(self.sound_path):
            if self.sound_path.lower().endswith('.wav'):
                winsound.PlaySound(self.sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                ps_cmd = f'(New-Object Media.SoundPlayer "{self.sound_path}").PlaySync()'
                os.system(f'powershell -c "{ps_cmd}"')
        else:
            phrase = "20 31 прибыл Сатору Годжо"
            ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{phrase}')"
            os.system(f'powershell -Command "{ps_cmd}"')
    
    def trigger(self):
        self.active = True
        self.end_time = time.time() + self.duration
        threading.Thread(target=self._play_sound, daemon=True).start()
    
    def _get_eye_data(self, rgb_frame, timestamp_ms, w, h):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]

        def get_single_eye_data(full_indices, corner_indices):
            full_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in full_indices])
            center = tuple(np.mean(full_pts, axis=0).astype(int))
            
            corner_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in corner_indices])
            p1, p2 = corner_pts[0], corner_pts[1]
            eye_width = np.linalg.norm(p1 - p2)
            
            eye_radius = int(eye_width * 0.20 )
            return center, max(eye_radius, 3)

        left_data = get_single_eye_data(self.LEFT_EYE_FULL, self.LEFT_EYE_CORNERS)
        right_data = get_single_eye_data(self.RIGHT_EYE_FULL, self.RIGHT_EYE_CORNERS)

        return left_data, right_data

    def apply(self, frame, timestamp_ms):
        if not self.active:
            return frame

        if time.time() > self.end_time:
            self.active = False
            return frame
        
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (40, 10, 30), -1)
        frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

        eyes_data = self._get_eye_data(rgb_frame, timestamp_ms, w, h)

        if eyes_data:
            glow_layer = frame.copy()

            for center, radius in eyes_data:
                cv2.circle(frame, center, radius, (255, 230, 90), -1, cv2.LINE_AA)
                
                cv2.circle(frame, center, int(radius * 0.75), (180, 70, 0), 1, cv2.LINE_AA)
                
                cv2.circle(frame, center, int(radius * 0.5), (255, 250, 160), 1, cv2.LINE_AA)
                
                cv2.circle(frame, center, max(int(radius * 0.25), 2), (255, 255, 255), -1, cv2.LINE_AA)

                glare_pos = (center[0] - max(int(radius * 0.2), 1), center[1] - max(int(radius * 0.2), 1))
                cv2.circle(frame, glare_pos, max(int(radius * 0.15), 1), (255, 255, 255), -1, cv2.LINE_AA)
            
        return frame