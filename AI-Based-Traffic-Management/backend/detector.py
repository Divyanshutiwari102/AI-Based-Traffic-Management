from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2 as cv
import numpy as np

from config import (
    CLASS_NAMES_PATH,
    DEFAULT_MAX_VIDEO_FRAMES,
    LEGACY_YOLO_CONFIG,
    LEGACY_YOLO_WEIGHTS,
    MODEL_PATH,
)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 0: "person"}
WEIGHTS = {"car": 1.0, "motorcycle": 0.5, "bus": 2.0, "truck": 1.5, "person": 0.3}

_YOLO_MODEL = None


def _get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    try:
        from ultralytics import YOLO  # type: ignore

        _YOLO_MODEL = YOLO(MODEL_PATH)
    except Exception:
        _YOLO_MODEL = None
    return _YOLO_MODEL


def weighted_density(counts: Dict[str, int], capacity: int) -> float:
    weighted = sum(WEIGHTS.get(cls_name, 1.0) * count for cls_name, count in counts.items())
    return min(weighted / max(capacity, 1), 1.0)


def lane_pressure(counts: Dict[str, int], capacity: int) -> float:
    return weighted_density(counts, capacity)


def detect_frame(frame: np.ndarray, conf: float = 0.4, imgsz: int = 640) -> Dict[str, int]:
    model = _get_yolo_model()
    if model is None:
        return {}

    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    counts: Counter[str] = Counter()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = VEHICLE_CLASSES.get(cls_id)
        if cls_name:
            counts[cls_name] += 1

    return dict(counts)


def _legacy_detect_cars(video_file: str) -> float:
    conf_threshold = 0.4
    nms_threshold = 0.4

    net = cv.dnn.readNet(LEGACY_YOLO_WEIGHTS, LEGACY_YOLO_CONFIG)
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    class_names = []
    class_file = Path(CLASS_NAMES_PATH)
    if not class_file.exists():
        return 0.0
    with class_file.open('r', encoding='utf-8') as file:
        class_names = [name.strip() for name in file.readlines()]

    cap = cv.VideoCapture(video_file)
    car_counts = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        classes, _, _ = model.detect(frame, conf_threshold, nms_threshold)
        car_count = 0
        for cls_id in classes:
            try:
                if class_names[int(cls_id)] == "car":
                    car_count += 1
            except Exception:
                continue
        car_counts.append(car_count)

    cap.release()
    if not car_counts:
        return 0.0
    return float(np.mean(car_counts))


def detect_video(video_file: str, max_frames: Optional[int] = DEFAULT_MAX_VIDEO_FRAMES) -> Dict[str, int]:
    cap = cv.VideoCapture(video_file)
    aggregate_counts: Counter[str] = Counter()
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counts = detect_frame(frame)
        if counts:
            aggregate_counts.update(counts)

        frame_index += 1
        if max_frames is not None and frame_index >= max_frames:
            break

    cap.release()

    if aggregate_counts:
        return dict(aggregate_counts)

    return {"car": int(round(_legacy_detect_cars(video_file)))}
