import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from inspector import features


def test_dinov2_checks_alignment_and_prevents_mismatched_local_weights(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="кратен 14"):
        features.Dinov2Embedder(image_size=225)

    monkeypatch.setenv("INSPECTOR_DINOV2_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="отсутствует"):
        features._local_dinov2_dir()

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Контрольная сумма"):
        features._local_dinov2_dir()


def test_dinov2_finds_checkout_weights_from_a_different_directory(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    model_dir = checkout / "artifacts" / "dinov2-small"
    model_dir.mkdir(parents=True)
    files = {"config.json": b"test config", "model.safetensors": b"test weights"}
    for name, content in files.items():
        (model_dir / name).write_bytes(content)
    monkeypatch.delenv("INSPECTOR_DINOV2_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(features, "__file__", str(checkout / "inspector" / "features.py"))
    monkeypatch.setattr(features, "DINOV2_SHA256", {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    })
    assert features._local_dinov2_dir() == model_dir


def test_dinov2_features_are_batched_and_keep_the_edges(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    loaded = []
    batches = []

    class TinyModel(torch.nn.Module):
        config = SimpleNamespace(hidden_size=2)

        def forward(self, pixel_values):
            batches.append(pixel_values.clone())
            means = pixel_values.mean(dim=(2, 3))
            return SimpleNamespace(last_hidden_state=torch.stack((means[:, :2], means[:, 1:]), dim=1))

    def load_model(*args, **kwargs):
        loaded.append((args, kwargs))
        return TinyModel()

    monkeypatch.setattr(transformers.Dinov2Model, "from_pretrained", load_model)
    monkeypatch.setattr(features, "_local_dinov2_dir", lambda: None)
    image = Image.new("RGB", (100, 20), color=(40, 80, 120))
    image.paste((220, 120, 50), (0, 0, 20, 20))
    images = [image, Image.new("L", (14, 20), color=75), image]

    embedder = features.Dinov2Embedder(image_size=28, batch_size=2)
    vector = embedder.embed(images)
    assert loaded == [((features.DINOV2_MODEL,), {
        "revision": features.DINOV2_REVISION, "use_safetensors": True,
    })]
    assert [tuple(batch.shape) for batch in batches] == [(2, 3, 28, 28), (1, 3, 28, 28)]
    assert batches[0][0, 0, 0, 0] > batches[0][0, 0, 0, -1]
    assert vector.dtype == np.float32
    assert vector.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(vector.reshape(3, 2, 2), axis=2), 1.0, atol=1e-6)
    np.testing.assert_allclose(vector[0], vector[2])
    assert embedder.embed([]).shape == (0, 4)

    hybrid, name, warning = features.get_embedder("dinov2_hybrid", image_size=28)
    assert name == "dinov2_hybrid"
    assert warning is None
    output = hybrid.embed(images)
    assert output.shape == (3, 4 + features.CLASSICAL_DIMENSION)
    np.testing.assert_allclose(output[:, :4], vector)
    np.testing.assert_allclose(output[:, 4:], features.ClassicalEmbedder(28).embed(images))
    assert hybrid.embed([]).shape == (0, output.shape[1])
