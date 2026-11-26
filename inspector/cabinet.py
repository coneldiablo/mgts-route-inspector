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
DEFAULT_CABINET_MODEL = Path("artifacts/final-cabinet/model.joblib")
ROI_RELEASE = "v0.3-roi"
CABINET_RELEASE = "v0.4-hr-roi"
SIGLIP_MODEL = "timm/vit_base_patch16_siglip_224.v2_webli"
SIGLIP_REVISION = "4c3661e5ac879a276ddc5ddc6d3f0ecc78fd5d82"
SIGLIP_SHA256 = "9106b0d8d9d02ea90fc3571fffd1557cf444736f695ee40b1e57c856bc3d9494"
DINOV2_BASE_MODEL = "facebook/dinov2-base"
DINOV2_BASE_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINOV2_BASE_SHA256 = {
    "config.json": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
    "model.safetensors": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
}


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


class Dinov2BaseEmbedder(Dinov2Embedder):
    def __init__(self, device: str = "cpu") -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Dinov2Model

        override = os.environ.get("INSPECTOR_DINOV2_BASE_DIR")
        if override:
            directory = Path(override).expanduser()
        else:
            relative = Path("artifacts/final-backbones/dinov2-base")
            candidates = (Path.cwd() / relative, Path(__file__).resolve().parents[1] / relative)
            directory = next((p for p in candidates if (p / "model.safetensors").is_file()), None)
            if directory is None:
                directory = Path(snapshot_download(
                    DINOV2_BASE_MODEL, revision=DINOV2_BASE_REVISION, allow_patterns=list(DINOV2_BASE_SHA256),
                ))
        for name, expected in DINOV2_BASE_SHA256.items():
            path = directory / name
            if not path.is_file() or image_sha256(path) != expected:
                raise ValueError(f"Не совпала контрольная сумма DINOv2-Base: {name}")
        self.model = Dinov2Model.from_pretrained(directory, local_files_only=True, use_safetensors=True)
        self.model.eval().to(device).requires_grad_(False)
        self.torch = torch
        self.device = device
        self.image_size = 224
        self.batch_size = 8
        self.embedding_dimension = 2 * self.model.config.hidden_size


class CabinetFeatures:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.dino: Dinov2Embedder | None = None
        self.base: Dinov2BaseEmbedder | None = None
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
        if variants & {"base", "base_multiscale", "base_siglip"}:
            if self.base is None:
                self.base = Dinov2BaseEmbedder(device=self.device)
            base_global = self.base.embed(images)
            if "base" in variants:
                features["base"] = base_global
            if "base_multiscale" in variants:
                local = [view for image in images for view in _views(image)]
                spatial = self.base.embed(local).reshape(len(images), 5, -1)
                corners = spatial[:, 1:]
                features["base_multiscale"] = np.concatenate(
                    (base_global, spatial[:, 0], corners.mean(axis=1), corners.std(axis=1)), axis=1,
                )
        if variants & {"siglip", "siglip_multi", "siglip_dino", "base_siglip"}:
            if self.siglip is None:
                self.siglip = SiglipEmbedder(device=self.device)
            use_crop = bool(variants & {"siglip_multi", "siglip_dino", "base_siglip"})
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
                if "base_siglip" in variants:
                    features["base_siglip"] = np.concatenate((base_global, paired), axis=1)
            else:
                features["siglip"] = self.siglip.embed(images)
        return features


def head_probabilities(specs: list[dict], features: dict[str, np.ndarray]) -> np.ndarray:
    probabilities = [spec["estimator"].predict_proba(features[spec["variant"]])[:, 1] for spec in specs]
    weights = [spec.get("weight", 1.0) for spec in specs]
    return np.average(probabilities, axis=0, weights=weights)


def blend_defect_scores(control: dict[str, float], hr: dict[str, float]) -> dict[str, float]:
    return control | {head: (control[head] + hr[head]) / 2 for head in HEADS[1:]}


def cabinet_result(scores: dict[str, float], quality, *, reported_type_score: float | None = None) -> dict:
    # The grounded crop can refine the displayed type without changing the action policy.
    policy_score = scores["telecom"]
    policy_confidence = max(policy_score, 1.0 - policy_score)
    final_score = policy_score if reported_type_score is None else reported_type_score
    object_index = int(final_score >= 0.5)
    object_confidence = max(final_score, 1.0 - final_score)
    signals = {head: round(float(scores[head]), 5) for head in HEADS[1:]}
    strongest = max(signals.values())
    if not quality.ok:
        action = "inspect"
        reason = "Требуется повторная съёмка или ручная проверка качества кадра."
    elif strongest >= 0.80 and policy_confidence >= 0.60:
        action = "repair"
        reason = "Выраженный визуальный признак. Мастер подтверждает объём ремонта."
    elif strongest > 0.30 or policy_confidence < 0.60:
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
        self.type_roi = artifact.get("type_roi")
        self.defect_hr = artifact.get("defect_hr")
        self.fallback_release = ROI_RELEASE if self.type_roi is not None else "v0.2"
        self.release = artifact.get("release", CABINET_RELEASE if self.defect_hr else self.fallback_release)
        self.grounder = None
        self.hr_encoder = None
        self.device = device
        self.features = CabinetFeatures(device)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CabinetInspector:
        return cls(joblib.load(path), device)

    def _cabinet_region(self, image: Image.Image):
        if self.type_roi is None:
            return None
        if self.grounder is None:
            from .cabinet_roi import CabinetGrounder, REVISION

            if self.type_roi.get("grounding_revision") != REVISION:
                raise ValueError("Версия детектора не соответствует классификатору области шкафа.")

            override = os.environ.get("INSPECTOR_CABINET_ROI_DIR")
            relative = Path("artifacts/cabinet-roi-grounding/detector")
            search = [Path(override).expanduser()] if override else [
                Path.cwd() / relative, Path(__file__).resolve().parents[1] / relative,
            ]
            required = ("model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json")
            directory = next((p for p in search if all((p / name).is_file() for name in required)), None)
            if directory is None:
                return None
            self.grounder = CabinetGrounder(directory, device=self.device)
        return self.grounder.locate(image)

    def _high_res_defects(self, image: Image.Image) -> dict[str, float] | None:
        if self.defect_hr is None:
            return None
        from scipy.special import expit
        from .cabinet_hr import DIM, HighResCabinetEncoder, REVISION, SHA256

        checkpoint = self.defect_hr
        if checkpoint.get("encoder_revision") != REVISION or checkpoint.get("encoder_sha256") != SHA256:
            raise ValueError("Версия SigLIP 2 HR не соответствует классификатору дефектов.")
        weights = checkpoint["weights"]
        bias = checkpoint["bias"]
        if weights.shape != (len(HEADS) - 1, DIM) or bias.shape != (len(HEADS) - 1,):
            raise ValueError("Некорректные размеры классификатора дефектов HR.")
        if self.hr_encoder is None:
            override = os.environ.get("INSPECTOR_SIGLIP_HR_DIR")
            relative = Path("artifacts/cabinet-hr-siglip")
            search = [Path(override).expanduser()] if override else [
                Path.cwd() / relative, Path(__file__).resolve().parents[1] / relative,
            ]
            directory = next((p for p in search if (p / "model.safetensors").is_file()), None)
            if directory is None:
                return None
            import torch

            auto = "mps" if torch.backends.mps.is_available() else "cpu"
            device = os.environ.get("INSPECTOR_HR_DEVICE", self.device if self.device != "cpu" else auto)
            self.hr_encoder = HighResCabinetEncoder(directory, device=device)
        probabilities = expit(weights @ self.hr_encoder.embed_image(image) + bias)
        return dict(zip(HEADS[1:], map(float, probabilities), strict=True))

    def predict_image(self, image: Image.Image) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        variants = {head["variant"] for specs in self.heads.values() for head in specs}
        region = self._cabinet_region(image)
        use_region = region is not None and region.bbox is not None
        if use_region:
            variants.update(("dino", "siglip"))
        vectors = self.features.extract([image], variants)
        scores = {
            name: float(head_probabilities(specs, vectors)[0])
            for name, specs in self.heads.items()
        }
        grounded_type = None
        if use_region:
            local = self.features.extract([region.view(image)], {"dino", "siglip"})
            combined = np.concatenate((vectors["dino"], vectors["siglip"], local["dino"], local["siglip"]), axis=1)
            grounded_type = float(self.type_roi["estimator"].predict_proba(combined)[0, 1])
        hr = self._high_res_defects(image)
        if hr is not None:
            scores = blend_defect_scores(scores, hr)
        result = cabinet_result(scores, assess_quality(image, bbox=None, cfg=quality_config("lowres")),
                                reported_type_score=grounded_type)
        result["object_type_source"] = "grounded_roi" if use_region else "full_frame"
        result["defect_score_source"] = "siglip_hr_blend" if hr is not None else "v02_ensemble"
        result["model_release"] = self.fallback_release if self.defect_hr is not None and hr is None else self.release
        if use_region:
            result["object_region"] = {"bbox": list(region.bbox), "score": region.score}
        return result

    def predict_path(self, path: str | Path) -> dict:
        return self.predict_image(open_image(path))
