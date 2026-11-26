"""Optional local manhole locator backed by an Ultralytics detector."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .detector import Detection


YOLO_LOCATOR_FORMAT = "mgts-yolo-locator-v1"
MANHOLE_CLASS_ID = 0


class YoloLocator:
    def __init__(self, predictor, image_size: int, score_threshold: float, model_name: str):
        self._predictor = predictor
        self._image_size = image_size
        self.score_threshold = score_threshold
        self.backend = model_name

    def locate_batch(self, images: list[Image.Image]) -> list[list[Detection]]:
        if not images:
            return []

        results = self._predictor.predict(
            source=[image.convert("RGB") for image in images],
            device="cpu",
            imgsz=self._image_size,
            conf=self.score_threshold,
            classes=[MANHOLE_CLASS_ID],
            max_det=20,
            verbose=False,
        )
        batches = []
        for image, result in zip(images, results, strict=True):
            detections: list[Detection] = []
            boxes = result.boxes
            if boxes is not None:
                coordinates = boxes.xyxy.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                for (x1, y1, x2, y2), score, label in zip(coordinates, scores, classes, strict=True):
                    if label != MANHOLE_CLASS_ID or not self.score_threshold <= score <= 1:
                        continue
                    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                        continue
                    box = (max(0.0, x1), max(0.0, y1), min(float(image.width), x2), min(float(image.height), y2))
                    if box[0] < box[2] and box[1] < box[3]:
                        detections.append((box, score))
            batches.append(sorted(detections, key=lambda detection: -detection[1]))
        return batches

    def locate(self, image: Image.Image) -> list[Detection]:
        return self.locate_batch([image])[0]


def load_yolo_locator(path: str | Path) -> YoloLocator:
    path = Path(path)
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Некорректный JSON локализатора YOLO.") from exc
    if not isinstance(descriptor, dict) or descriptor.get("format") != YOLO_LOCATOR_FORMAT:
        raise ValueError("Неизвестный формат артефакта локализации YOLO.")

    weights_name = descriptor.get("weights")
    image_size = descriptor.get("image_size")
    score_threshold = descriptor.get("score_threshold")
    model_name = descriptor.get("model")
    if not isinstance(weights_name, str) or not weights_name.strip():
        raise ValueError("Укажите относительный путь к весам локализатора YOLO.")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("Размер кадра локализатора YOLO должен быть положительным целым числом.")
    if type(score_threshold) not in (float, int) or not 0 <= score_threshold <= 1:
        raise ValueError("Порог локализации YOLO должен быть в диапазоне [0, 1].")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Укажите название модели локализатора YOLO.")

    relative = Path(weights_name)
    root = path.parent.resolve()
    weights = (root / relative).resolve()
    if relative.is_absolute() or not weights.is_relative_to(root) or weights.suffix.lower() != ".pt":
        raise ValueError("Веса локализатора YOLO должны быть файлом .pt в каталоге дескриптора.")
    if not weights.is_file():
        raise FileNotFoundError(f"Веса локализатора YOLO не найдены: {weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Для локализатора YOLO установите дополнительный пакет ultralytics.") from exc

    model = YOLO(str(weights), task="detect", verbose=False)
    return YoloLocator(model, image_size, float(score_threshold), model_name.strip())
