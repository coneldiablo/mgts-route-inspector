"""Feature extractors for the frozen MobileNet and DINOv2 backbones and classical CV.

The CNN branch uses ImageNet weights and a light logistic head. It does not
fine-tune the backbone on the hackathon laptop, which keeps cold start and
training time low and is robust when only a few hundred labelled frames exist.
The classical branch is deliberately simple; it is both the comparison baseline
and an extra feature block in the default hybrid variant.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSICAL_DIMENSION = 222
DINOV2_MODEL = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DINOV2_SHA256 = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
}


class TorchEmbedder:
    name = "mobilenet_v3_small"

    def __init__(self, device: str | None = None, image_size: int = 224, batch_size: int = 16):
        import torch
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval().to(device)
        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                chunk = images[start : start + self.batch_size]
                batch = self.torch.stack([self.transform(im.convert("RGB")) for im in chunk]).to(self.device)
                out = self.model(batch)
                vectors.append(out.detach().float().cpu().numpy())
        if not vectors:
            return np.zeros((0, 576), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


def _local_dinov2_dir() -> Path | None:
    override = os.environ.get("INSPECTOR_DINOV2_DIR")
    if override:
        directory = Path(override).expanduser()
    else:
        relative = Path("artifacts", "dinov2-small")
        roots = (Path.cwd(), Path(__file__).resolve().parents[1])
        directory = next((root / relative for root in roots if (root / relative).is_dir()), None)
        if directory is None:
            return None

    for name, expected in DINOV2_SHA256.items():
        file = directory / name
        if not file.is_file():
            raise ValueError(f"В каталоге DINOv2 отсутствует {file}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Контрольная сумма файла {file} не совпадает с официальной версией DINOv2")
    return directory


class Dinov2Embedder:
    """Frozen visual features from the CLS token and the mean of the image patch tokens."""

    name = "dinov2_small"

    def __init__(self, device: str = "cpu", image_size: int = 224, batch_size: int = 8):
        if image_size < 14 or image_size % 14:
            raise ValueError("Размер изображения для DINOv2 должен быть кратен 14")
        if batch_size < 1:
            raise ValueError("Размер пакета должен быть положительным")
        try:
            import torch
            from transformers import Dinov2Model
        except ImportError as exc:
            raise ValueError("Установите PyTorch и Transformers для DINOv2") from exc

        directory = _local_dinov2_dir()
        if directory is None:
            model = Dinov2Model.from_pretrained(DINOV2_MODEL, revision=DINOV2_REVISION, use_safetensors=True)
        else:
            model = Dinov2Model.from_pretrained(directory, local_files_only=True, use_safetensors=True)
        self.model = model.eval().to(device)
        self.model.requires_grad_(False)
        self.torch = torch
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.embedding_dimension = 2 * model.config.hidden_size

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                chunk = images[start : start + self.batch_size]
                # Resize the entire image or supplied object crop. A centre crop
                # here could remove road assets at the edge of the frame.
                pixels = np.stack([
                    np.asarray(image.convert("RGB").resize(
                        (self.image_size, self.image_size), Image.Resampling.BICUBIC
                    ), dtype=np.float32)
                    for image in chunk
                ])
                pixels /= 255.0
                pixels -= IMAGENET_MEAN
                pixels /= IMAGENET_STD
                batch = self.torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous().to(self.device)
                tokens = self.model(pixel_values=batch).last_hidden_state.float()
                cls = self.torch.nn.functional.normalize(tokens[:, 0], dim=1)
                patches = self.torch.nn.functional.normalize(tokens[:, 1:].mean(dim=1), dim=1)
                vectors.append(self.torch.cat((cls, patches), dim=1).cpu().numpy())
        if not vectors:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


class ClassicalEmbedder:
    """Small, dependency-light baseline based on colour, texture and geometry."""

    name = "classical_cv"
    def __init__(self, size: int = 128):
        self.size = size

    @staticmethod
    def _hist(values: np.ndarray, bins: int) -> np.ndarray:
        h, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
        h = h.astype(np.float32)
        return h / max(1.0, h.sum())

    def _one(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        feats: list[np.ndarray] = []

        for c in range(3):
            feats.append(self._hist(rgb[:, :, c], 12))
        for c in range(3):
            feats.append(self._hist(hsv[:, :, c], 10))

        # Gradient orientation histogram on a 4x4 grid.
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        bins = np.minimum((ang * 8).astype(np.int32), 7)
        cell = self.size // 4
        for cy in range(4):
            for cx in range(4):
                sl = (slice(cy * cell, (cy + 1) * cell), slice(cx * cell, (cx + 1) * cell))
                h = np.bincount(
                    bins[sl].ravel(),
                    weights=mag[sl].ravel(),
                    minlength=8,
                ).astype(np.float32)
                feats.append(h / max(1e-6, h.sum()))

        # Radial profile around the frame centre (manhole covers are round).
        yy, xx = np.mgrid[0 : self.size, 0 : self.size]
        cy = cx = (self.size - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (self.size / 2.0)
        for r0 in np.linspace(0.0, 0.9, 10):
            mask = (radius >= r0) & (radius < r0 + 0.1)
            if mask.any():
                feats.append(np.array([gray[mask].mean(), gray[mask].std()], dtype=np.float32))
            else:
                feats.append(np.zeros(2, dtype=np.float32))

        # Centre vs border contrast, dark-area fraction, edge density.
        centre_mask = radius < 0.35
        border_mask = radius > 0.75
        centre_mean = float(gray[centre_mask].mean())
        border_mean = float(gray[border_mask].mean())
        feats.append(
            np.array(
                [
                    centre_mean,
                    border_mean,
                    centre_mean - border_mean,
                    float((gray < 0.18).mean()),
                    float((gray < 0.30).mean()),
                    float(mag.mean()),
                    float(np.percentile(mag, 90)),
                    float((mag > 0.12).mean()),
                ],
                dtype=np.float32,
            )
        )
        return np.concatenate(feats).astype(np.float32)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._one(im) for im in images])


class HybridEmbedder:
    """CNN semantics plus explicit colour/texture/geometry features.

    On small labelled sets the CNN embedding can miss simple cues (e.g. the dark
    opening inside a round cover). The union keeps ImageNet transfer while giving
    the linear head a few interpretable measurements. Both parts are fitted on
    the same frames; the scaler handles different scales.
    """

    name = "hybrid_cnn_cv"

    def __init__(self, image_size: int = 224):
        self.cnn = TorchEmbedder(image_size=image_size)
        self.classical = ClassicalEmbedder(image_size)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 576 + CLASSICAL_DIMENSION), dtype=np.float32)
        return np.hstack([self.cnn.embed(images), self.classical.embed(images)]).astype(np.float32)


class Dinov2HybridEmbedder:
    """DINOv2 visual features combined with the existing colour and texture features."""

    name = "dinov2_hybrid"

    def __init__(self, image_size: int = 224):
        self.vision = Dinov2Embedder(image_size=image_size)
        self.classical = ClassicalEmbedder(image_size)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.vision.embedding_dimension + CLASSICAL_DIMENSION), dtype=np.float32)
        return np.hstack([self.vision.embed(images), self.classical.embed(images)])


def get_embedder(backbone: str, image_size: int = 224):
    """Return (embedder, actual_name, warning)."""
    if backbone in {"classical", "classical_cv"}:
        return ClassicalEmbedder(image_size), "classical_cv", None
    if backbone in {"hybrid", "hybrid_cnn_cv", "mobilenet_classical", "cnn_classical"}:
        return HybridEmbedder(image_size=image_size), HybridEmbedder.name, None
    if backbone == "auto":
        try:
            return HybridEmbedder(image_size=image_size), HybridEmbedder.name, None
        except Exception as exc:  # missing torch, no weights, no network
            warning = (
                "torch/torchvision недоступны или веса не загрузились; "
                f"использую классические признаки. Причина: {exc}"
            )
            warnings.warn(warning, RuntimeWarning)
            return ClassicalEmbedder(image_size), "classical_cv", warning
    if backbone in {"mobilenet_v3_small", "mobilenet"}:
        return TorchEmbedder(image_size=image_size), "mobilenet_v3_small", None
    if backbone == "dinov2_small":
        return Dinov2Embedder(image_size=image_size), Dinov2Embedder.name, None
    if backbone == "dinov2_hybrid":
        return Dinov2HybridEmbedder(image_size=image_size), Dinov2HybridEmbedder.name, None
    raise ValueError(f"Неизвестный backbone: {backbone}")
