import sys
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction, QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QSystemTrayIcon, QMenu
)


import action
from gesture import process_gesture

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def create_status_icon(color_hex):
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color_hex))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 28, 28)
    painter.end()
    return QIcon(pixmap)

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    status_changed_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._is_running = True
        self.last_status = None
        self.cap = None

    def run(self):
        model_path = os.path.join('models', 'hand_landmarker.task')

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1
        )

        self.cap = cv2.VideoCapture(0)

        with HandLandmarker.create_from_options(options) as landmarker:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret:
                    break

                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                frame_timestamp_ms = int(time.time() * 1000)

                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                gesture_name = "Unknown"
                confidence = 0.0
                
                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        h, w, _ = frame.shape
                        
                        for connection in HAND_CONNECTIONS:
                            start, end = connection
                            pt1 = (int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h))
                            pt2 = (int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                        for lm in hand_landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                        gesture_name, confidence = process_gesture(hand_landmarks, frame, frame_timestamp_ms)
                else:
                    process_gesture(None, frame, frame_timestamp_ms)
                        
                current_status = action.actions_enabled
                if current_status != self.last_status:
                    self.last_status = current_status
                    self.status_changed_signal.emit(current_status)

                status_text = "[ACTIVE]" if action.actions_enabled else "[MUTED]"
                cv2.putText(frame, f"Жест: {gesture_name} ({confidence * 100:.1f}%) {status_text}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.change_pixmap_signal.emit(frame)

        if self.cap and self.cap.isOpened():
            self.cap.release()

    def stop(self):
        self._is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.wait(200)

class CustomTitleBar(QWidget):
    """
    Создания заголовка с возможностью как и закрыть, свернуть так и закрыть в трей
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.drag_position = QPoint()

        self.setFixedHeight(35)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #ff1744; font-size: 14px; margin-right: 6px;")
        layout.addWidget(self.status_dot)

        self.title_label = QLabel("Mizim's Gesture")
        self.title_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(self.title_label)

        layout.addStretch()

        btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                color: #cccccc;
                font-size: 13px;
                width: 40px;
                height: 35px;
            }
            QPushButton:hover {
                background-color: #333333;
                color: #ffffff;
            }
        """

        self.btn_tray = QPushButton("/")
        self.btn_tray.setToolTip("Свернуть в трей")
        self.btn_tray.setStyleSheet(btn_style)
        self.btn_tray.clicked.connect(self.parent.hide)
        layout.addWidget(self.btn_tray)

        self.btn_min = QPushButton("🗕")
        self.btn_min.setToolTip("Свернуть")
        self.btn_min.setStyleSheet(btn_style)
        self.btn_min.clicked.connect(self.parent.showMinimized)
        layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("🗖")
        self.btn_max.setToolTip("На весь экран")
        self.btn_max.setStyleSheet(btn_style)
        self.btn_max.clicked.connect(self.toggle_max)
        layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setToolTip("Закрыть")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #cccccc;
                font-size: 13px;
                width: 40px;
                height: 35px;
            }
            QPushButton:hover {
                background-color: #e81123;
                color: #ffffff;
            }
        """)
        self.btn_close.clicked.connect(self.parent.close)
        layout.addWidget(self.btn_close)

    def toggle_max(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.parent.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.parent.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
    
class GestureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: #121212;")
        self.resize(720, 520)

        self.icon_active = create_status_icon("#00e676")
        self.icon_muted = create_status_icon("#ff1744")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self)
        main_layout.addWidget(self.title_bar)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        main_layout.addWidget(self.image_label)

        self.setup_tray()

        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_changed_signal.connect(self.update_status_display)
        self.thread.start()
    
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def update_status_display(self, is_active):
        if is_active:
            current_icon = self.icon_active
            status_text = "Mizim's Gesture: [ACTIVE]"
            self.title_bar.status_dot.setStyleSheet("color: #00e676; font-size: 14px; margin-right: 6px;")
        else:
            current_icon = self.icon_muted
            status_text = "Mizim's Gesture: [MUTED]"
            self.title_bar.status_dot.setStyleSheet("color: #ff1744; font-size: 14px; margin-right: 6px;")

        self.setWindowIcon(current_icon)
        QApplication.setWindowIcon(current_icon)
        self.tray_icon.setIcon(current_icon)
        self.tray_icon.setToolTip(status_text)
    
    def setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.icon_muted)
        self.tray_icon.setToolTip("Mizim's Gesture: ВЫКЛЮЧЕНО")

        tray_menu = QMenu()
        
        show_action = QAction("Открыть панель", self)
        show_action.triggered.connect(self.showNormal)
        tray_menu.addAction(show_action)

        tray_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close_app)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()
    
    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.showNormal()
    
    def closeEvent(self, event):
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            if self.thread.isRunning():
                self.thread.terminate()
        event.accept()
        QApplication.quit()

    def close_app(self):
        self.close()
