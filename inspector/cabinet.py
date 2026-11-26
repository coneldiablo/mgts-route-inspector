"""Local cabinet inference with independent object, mechanical and paint heads."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from PIL import Image, ImageOps

from .config import quality_config
from .features import ClassicalEmbedder, Dinov2Embedder, TorchEmbedder
from .manifest import image_sha256
from .quality import assess_quality, open_image

HEADS = ("telecom", "mechanical", "coating")
OBJECTS = ("distribution_board", "telecom_cabinet")
MODEL_FORMAT = 1
SIGLIP_MODEL = "timm/vit_base_patch16_siglip_224.v2_webli"
SIGLIP_REVISION = "4c3661e5ac879a276ddc5ddc6d3f0ecc78fd5d82"
SIGLIP_SHA256 = "9106b0d8d9d02ea90fc3571fffd1557cf444736f695ee40b1e57c856bc3d9494"


def _views(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    windows = ((.15, .15, .85, .85), (0, 0, .68, .68), (.32, 0, 1, .68),
               (0, .32, .68, 1), (.32, .32, 1, 1))
    if min(width, height) < 8:
        return [image] * len(windows)
    return [
        image.crop((round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)))
        for x0, y0, x1, y1 in windows
    ]


class SiglipEmbedder:
    def __init__(self, device: str = "cpu") -> None:
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        override = os.environ.get("INSPECTOR_SIGLIP_DIR")
        candidate = Path(override).expanduser() if override else Path("artifacts/final-backbones/siglip2")
        path = candidate / "model.safetensors"
        if not path.is_file() and not override:
            path = Path(hf_hub_download(SIGLIP_MODEL, "model.safetensors", revision=SIGLIP_REVISION))
        if image_sha256(path) != SIGLIP_SHA256:
            raise ValueError("Контрольная сумма модели SigLIP 2 не совпадает с указанной версией.")
        self.torch = torch
        self.device = device
        self.model = timm.create_model(
            "vit_base_patch16_siglip_224.v2_webli", pretrained=False, num_classes=0, checkpoint_path=str(path),
        ).eval().to(device)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self.torch
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(images), 8):
                pixels = np.stack([
                    np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC))
                    for image in images[start:start + 8]
                ]).astype(np.float32)
                batch = torch.from_numpy(pixels / 127.5 - 1).permute(0, 3, 1, 2).contiguous().to(self.device)
                vectors.append(torch.nn.functional.normalize(self.model(batch), dim=1).cpu().numpy())
        return np.concatenate(vectors)


class CabinetFeatures:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.dino: Dinov2Embedder | None = None
        self.mobile: TorchEmbedder | None = None
        self.siglip: SiglipEmbedder | None = None

    def extract(self, images: Sequence[Image.Image], variants: set[str]) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {}
        cv = ClassicalEmbedder().embed(images) if variants & {"classical", "hybrid"} else None
        if "classical" in variants:
            features["classical"] = cv
        if variants & {"dino", "multiscale", "hybrid", "siglip_dino"}:
            if self.dino is None:
                self.dino = Dinov2Embedder(device=self.device)
            global_vectors = self.dino.embed(images)
            if "dino" in variants:
                features["dino"] = global_vectors
            if variants & {"multiscale", "hybrid"}:
                local_views = [view for image in images for view in _views(image)]
                vectors = self.dino.embed(local_views).reshape(len(images), 5, -1)
                pooled = np.concatenate((global_vectors, vectors.mean(axis=1), vectors[:, 0]), axis=1)
                if "multiscale" in variants:
                    features["multiscale"] = pooled
                if "hybrid" in variants:
                    features["hybrid"] = np.concatenate((pooled, cv), axis=1)
        if "mobile" in variants:
            if self.mobile is None:
                self.mobile = TorchEmbedder(device=self.device)
            features["mobile"] = self.mobile.embed(images)
        if variants & {"siglip", "siglip_multi", "siglip_dino"}:
            if self.siglip is None:
                self.siglip = SiglipEmbedder(device=self.device)
            use_crop = bool(variants & {"siglip_multi", "siglip_dino"})
            if use_crop:
                views = [view for image in images for view in (image, _views(image)[0])]
                vectors = self.siglip.embed(views).reshape(len(images), 2, -1)
                if "siglip" in variants:
                    features["siglip"] = vectors[:, 0]
                paired = vectors.reshape(len(images), -1)
                if "siglip_multi" in variants:
                    features["siglip_multi"] = paired
                if "siglip_dino" in variants:
                    features["siglip_dino"] = np.concatenate((global_vectors, paired), axis=1)
            else:
                features["siglip"] = self.siglip.embed(images)
        return features


def cabinet_result(scores: dict[str, float], quality) -> dict:
    object_index = int(scores["telecom"] >= 0.5)
    object_confidence = scores["telecom"] if object_index else 1.0 - scores["telecom"]
    signals = {head: round(float(scores[head]), 5) for head in HEADS[1:]}
    strongest = max(signals.values())
    if not quality.ok:
        action = "inspect"
        reason = "Требуется повторная съёмка или ручная проверка качества кадра."
    elif strongest >= 0.80 and object_confidence >= 0.60:
        action = "repair"
        reason = "Выраженный визуальный признак. Мастер подтверждает объём ремонта."
    elif strongest > 0.30 or object_confidence < 0.60:
        action = "inspect"
        reason = "Мастер проверяет признаки и тип объекта."
    else:
        action = "no_action"
        reason = "Видимые признаки механического повреждения и дефекта покрытия не выявлены."
    findings = [{"code": name, "score": score} for name, score in signals.items() if score >= 0.50]
    return {
        "object_type": OBJECTS[object_index],
        "object_type_score": round(float(object_confidence), 5),
        "scores": signals,
        "findings": findings,
        "action": action,
        "reason": reason,
        "quality": quality.to_dict(),
    }


class CabinetInspector:
    """Image-only predictions, followed by a separately configurable action policy."""

    def __init__(self, artifact: dict, device: str = "cpu") -> None:
        if artifact.get("format") != MODEL_FORMAT or set(artifact.get("heads", ())) != set(HEADS):
            raise ValueError("Неподдерживаемый формат модели шкафов.")
        self.heads = {name: spec if isinstance(spec, list) else [spec] for name, spec in artifact["heads"].items()}
        self.features = CabinetFeatures(device)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CabinetInspector:
        return cls(joblib.load(path), device)

    def predict_image(self, image: Image.Image) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        variants = {head["variant"] for specs in self.heads.values() for head in specs}
        vectors = self.features.extract([image], variants)
        scores = {
            name: float(np.mean([head["estimator"].predict_proba(vectors[head["variant"]])[0, 1]
                                 for head in specs]))
            for name, specs in self.heads.items()
        }
        return cabinet_result(scores, assess_quality(image, bbox=None, cfg=quality_config("lowres")))

    def predict_path(self, path: str | Path) -> dict:
        return self.predict_image(open_image(path))
