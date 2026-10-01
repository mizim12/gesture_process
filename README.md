# Gesture Process

Приложение на Python для управления ПК с помощью жестов рук в реальном времени. Построено на PyQt6, OpenCV и MediaPipe.

---

## Возможности

- Распознавание жестов: отслеживание рук через веб-камеру с помощью MediaPipe Landmarker.
- Управление состояниями: быстрые режимы [ACTIVE] / [MUTED].
- Системный трей: фоновый режим работы, динамическая иконка в трее и уведомления.
- Спецэффекты: визуальные надстройки (Domain Expansion / Six Eyes) с использованием детекции лица.
- Обучение: скрипты для сбора собственного датасета и обучения ML-модели.

---

## Структура проекта

```text
Gesture/
├── assets/                   # Медиафайлы и видеоэффекты
├── dataset/                  # Сбор данных и обучение (collect_dataset.py, train_model.py)
├── models/                   # Модели MediaPipe и обученная gesture_model.pkl
├── action.py                 # Логика выполнения команд
├── domain_expansion.py       # Визуальные эффекты
```

## Установка зависимостей

```bash
pip install opencv-python mediapipe PyQt6 scikit-learn numpy
```

## Запуск проекта

```bash
python main.py
```


├── gesture.py                # Классификация жестов
├── gui.py                    # Интерфейс PyQt6 и поток камеры
└── main.py                   # Точка входа
