"""Frozen SigLIP 2/384 vision features from overlapping views of a cabinet."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from inspector.manifest import image_sha256

ARCH = "vit_so400m_patch16_siglip_384.v2_webli"
MODEL = "timm/" + ARCH
REVISION = "5269a45a3c4b03a8efae511eac9bcacdf4bf55e9"
SHA256 = "542bac8ad54c1effc887315d3d6448fed01a26c900064bc47f342ed5e3e5fb62"
FRACTION = .72
SIDE = 384
DIM = 1152


class HighResCabinetEncoder:
    def __init__(self, directory: str | Path | None = None, device: str = "cpu"):
        import torch
        import timm
        from huggingface_hub import hf_hub_download

        directory = directory or os.environ.get("INSPECTOR_SIGLIP_HR_DIR")
        if directory is None:
            cached = Path(__file__).resolve().parents[1] / "artifacts/cabinet-hr-siglip/model.safetensors"
            weights = cached if cached.exists() else Path(hf_hub_download(MODEL, "model.safetensors", revision=REVISION))
        else:
            weights = Path(directory) / "model.safetensors"
        if image_sha256(weights) != SHA256:
            raise ValueError("High-resolution SigLIP 2 weights failed checksum verification.")
        self.device = device
        self.dtype = torch.float16 if device in {"mps","cuda"} else torch.float32
        self.torch = torch
        self.model = timm.create_model(ARCH, pretrained=False, num_classes=0, checkpoint_path=str(weights))
        self.model.eval().to(device=device, dtype=self.dtype)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        width, height = image.size
        cw, ch = max(1, round(width * FRACTION)), max(1, round(height * FRACTION))
        offsets = [(0, 0), (width - cw, 0), (0, height - ch), (width - cw, height - ch)]
        vectors = []
        torch = self.torch
        with torch.inference_mode():
            for i in range(0, len(offsets), 2):
                view_pixels = []
                for x,y in offsets[i:i+2]:
                    crop = image.crop((x,y,x+cw,y+ch)).resize((SIDE, SIDE), Image.Resampling.BICUBIC)
                    view_pixels.append(np.asarray(crop, dtype=np.float32) / 127.5 - 1)
                batch = torch.from_numpy(np.stack(view_pixels)).permute(0,3,1,2).contiguous()
                output = self.model(batch.to(device=self.device, dtype=self.dtype)).float()
                vectors.extend(torch.nn.functional.normalize(output, dim=1).cpu().numpy())
        result = np.mean(vectors, axis=0)
        if result.shape != (DIM,) or not np.all(np.isfinite(result)):
            raise ValueError("Invalid high-resolution cabinet embedding.")
        return result


class HighResCabinetScorer:
    def __init__(self, head: str | Path, directory: str | Path | None = None, device: str = "cpu"):
        with np.load(head, allow_pickle=False) as data:
            self.weights = data["weights"].copy()
            self.bias = data["bias"].copy()
        if self.weights.shape != (3, DIM) or self.bias.shape != (3,):
            raise ValueError("Invalid high-resolution cabinet heads.")
        self.encoder = HighResCabinetEncoder(directory, device)

    def scores(self, image: Image.Image) -> dict[str,float]:
        from scipy.special import expit
        values = expit(self.weights @ self.encoder.embed_image(image) + self.bias)
        return {key: float(v) for key,v in zip(("telecom","mechanical","coating"), values, strict=True)}
