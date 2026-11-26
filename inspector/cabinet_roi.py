"""Text-grounded cabinet crop for an optional, local second view of field images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from inspector.manifest import image_sha256
from inspector.quality import crop_bbox

MODEL = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
WEIGHTS_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
PROMPT = "electrical cabinet. utility box. metal cabinet."
MIN_SCORE = 0.30
MIN_AREA = 0.02
MARGIN = 0.08


@dataclass(frozen=True)
class CabinetRegion:
    """Normalized rectangle before context padding; empty means use the complete image."""
    bbox: tuple[float, float, float, float] | None
    score: float | None

    def view(self, image: Image.Image) -> Image.Image:
        if self.bbox is None:
            return image
        width, height = image.size
        x1, y1, x2, y2 = self.bbox
        return crop_bbox(image, (x1 * width, y1 * height, x2 * width, y2 * height), MARGIN)


class CabinetGrounder:
    def __init__(self, directory: str | Path | None = None, device: str = "cpu"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if directory is None:
            directory = snapshot_download(
                MODEL, revision=REVISION,
                allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json",
                                "tokenizer*.json", "added_tokens.json", "special_tokens_map.json", "vocab.txt"],
            )
        directory = Path(directory)
        if image_sha256(directory / "model.safetensors") != WEIGHTS_SHA256:
            raise ValueError("Incorrect Grounding DINO checkpoint checksum.")
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            directory, local_files_only=True, use_safetensors=True,
        ).eval().to(device)
        self.device = device
        self.torch = torch

    def locate(self, image: Image.Image) -> CabinetRegion:
        batch = self.processor(
            images=image, text=PROMPT, return_tensors="pt",
            size={"shortest_edge": 560, "longest_edge": 900},
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**batch)
        detections = self.processor.post_process_grounded_object_detection(
            outputs, batch.input_ids, threshold=MIN_SCORE, text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]
        best = CabinetRegion(None, None)
        for box, score in zip(detections["boxes"].tolist(), detections["scores"].tolist(), strict=True):
            x1, y1, x2, y2 = (max(0.0, min(1.0, value / size)) for value, size in
                              zip(box, (image.width, image.height, image.width, image.height), strict=True))
            area = (x2 - x1) * (y2 - y1)
            if x2 > x1 and y2 > y1 and area >= MIN_AREA and (best.score is None or score > best.score):
                best = CabinetRegion((x1, y1, x2, y2), float(score))
        return best


class GroundedCabinetTyper:
    """Optional object-type head using the complete image and a proposed cabinet crop."""

    def __init__(self, head: str | Path, detector: str | Path | None = None, device: str = "cpu"):
        import numpy as np
        from inspector.cabinet import SiglipEmbedder
        from inspector.features import Dinov2Embedder

        with np.load(head, allow_pickle=False) as data:
            self.weights = data["weights"].copy()
            self.bias = float(data["bias"].item())
        if self.weights.shape != (3072,) or not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise ValueError("Invalid grounded cabinet type head.")
        self.grounder = CabinetGrounder(detector, device=device)
        self.dino = Dinov2Embedder(device=device)
        self.siglip = SiglipEmbedder(device=device)

    def predict_image(self, image: Image.Image) -> dict:
        import numpy as np
        from PIL import ImageOps
        from scipy.special import expit

        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        region = self.grounder.locate(image)
        views = [image, region.view(image)]
        dino = self.dino.embed(views)
        siglip = self.siglip.embed(views)
        features = np.concatenate((dino[0], siglip[0], dino[1], siglip[1]))
        telecom_score = float(expit(self.bias + features @ self.weights))
        return {
            "object_type": "telecom_cabinet" if telecom_score >= .50 else "distribution_board",
            "telecom_score": telecom_score, "grounding_score": region.score, "cabinet_box": region.bbox,
        }
