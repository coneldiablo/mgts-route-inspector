"""Manhole bounding boxes on full frames and locator artifact loading."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from PIL import Image

ARCHITECTURE = "ssdlite320_mobilenet_v3_large_manhole"
LOCATOR_FORMAT = 1
Box = tuple[float, float, float, float]
Detection = tuple[Box, float]


class Locator(Protocol):
    backend: str
    score_threshold: float

    def locate_batch(self, images: list[Image.Image]) -> list[list[Detection]]: ...


def build_locator_model(pretrained: bool = False):
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    options = dict(score_thresh=0.01, detections_per_img=20, topk_candidates=100)
    if not pretrained:
        # The COCO detector also has this reduced backbone tail. Reconstruct its
        # exact layout here without downloading anything at inference time.
        return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2, **options)

    from functools import partial

    from torch import nn
    from torchvision.models.detection._utils import retrieve_out_channels
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1, trainable_backbone_layers=0, **options,
    )
    model.backbone.requires_grad_(False)
    classification = SSDLiteClassificationHead(
        retrieve_out_channels(model.backbone, (320, 320)),
        model.anchor_generator.num_anchors_per_location(),
        num_classes=2,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
    )
    for new, old in zip(classification.module_list, model.head.classification_head.module_list, strict=True):
        new[0].load_state_dict(old[0].state_dict())
    model.head.classification_head = classification
    return model


def box_iou(one: Box, two: Box) -> float:
    width = max(0.0, min(one[2], two[2]) - max(one[0], two[0]))
    height = max(0.0, min(one[3], two[3]) - max(one[1], two[1]))
    intersection = width * height
    union = (one[2] - one[0]) * (one[3] - one[1]) + (two[2] - two[0]) * (two[3] - two[1]) - intersection
    return intersection / union if union > 0 else 0.0


class ManholeLocator:
    backend = "SSDLite"

    def __init__(self, model, score_threshold: float, device: str = "cpu"):
        if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError("Порог локализации должен быть в диапазоне [0, 1].")
        self._model = model
        self._device = device
        self.score_threshold = score_threshold

    def locate_batch(self, images: list[Image.Image]) -> list[list[Detection]]:
        import torch
        from torchvision.transforms.functional import to_tensor

        if not images:
            return []
        self._model.eval()
        with torch.inference_mode():
            predictions = self._model([to_tensor(img.convert("RGB")).to(self._device) for img in images])
        batches = []
        for image, prediction in zip(images, predictions, strict=True):
            boxes = prediction["boxes"].cpu().tolist()
            scores = prediction["scores"].cpu().tolist()
            labels = prediction["labels"].cpu().tolist()
            detections: list[Detection] = []
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels, strict=True):
                if label != 1 or score < self.score_threshold:
                    continue
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
                    continue
                box = (max(0.0, x1), max(0.0, y1), min(float(image.width), x2), min(float(image.height), y2))
                if box[0] < box[2] and box[1] < box[3]:
                    detections.append((box, score))
            batches.append(sorted(detections, key=lambda detection: -detection[1]))
        return batches

    def locate(self, image: Image.Image) -> list[Detection]:
        return self.locate_batch([image])[0]


def load_locator(path: str | Path) -> Locator:
    path = Path(path)
    if path.suffix.lower() == ".json":
        from .yolo_locator import load_yolo_locator

        return load_yolo_locator(path)

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format") != LOCATOR_FORMAT
            or payload.get("architecture") != ARCHITECTURE):
        raise ValueError("Неизвестный формат артефакта локализации.")
    model = build_locator_model()
    model.load_state_dict(payload["state_dict"])
    return ManholeLocator(model.eval(), float(payload["score_threshold"]))
