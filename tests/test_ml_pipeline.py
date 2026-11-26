import csv
import json

import numpy as np
import pytest
from PIL import Image

from inspector.classifier import Prediction, load_model, train_model
from inspector.config import AppConfig, ModelConfig, QualityConfig
from inspector.evaluate import evaluate_artifact, evaluate_holdout
from inspector.manifest import Record, load_manifest
from inspector.pipeline import predict_records
from scripts.benchmark_vertex import decode_response, request_body, summarize


def _make_damage_pack(root, count, seed):
    root.mkdir()
    rng = np.random.default_rng(seed)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["image", "asset_id", "label"])
        writer.writeheader()
        for i in range(count):
            label = ("broken", "shifted")[i % 2]
            name = f"{seed}-{i}.png"
            rgb = rng.integers(55, 95, (96, 112, 3), dtype=np.uint8)
            rgb[..., i % 2] += 90
            Image.fromarray(rgb).save(root / name)
            writer.writerow({"image": name, "asset_id": f"asset-{seed}-{i}", "label": label})


def _quality():
    return QualityConfig(min_width=80, min_height=80, min_sharpness=0.0, min_contrast=0.0)


def test_known_damage_mode_uses_multiclass_metrics_and_never_marks_normal(tmp_path):
    train, holdout, artifact = [tmp_path / name for name in ("train", "holdout", "model")]
    _make_damage_pack(train, 12, 1)
    _make_damage_pack(holdout, 6, 2)
    config = ModelConfig(backbone="classical", defect_only=True, cv_folds=3)
    meta = train_model(train, artifact, cfg=config, quality=_quality(), verbose=False)
    assert meta["mode"] == "defect_type"
    assert meta["threshold_mode"] == "type_macro_f1"
    assert evaluate_artifact(artifact, train)["n_images"] == 12
    holdout_result = evaluate_holdout(artifact, holdout)
    assert holdout_result["n_images"] == 6
    assert holdout_result["type_macro_f1"] > 0.9
    assert set(holdout_result["per_class"]) == {"broken", "shifted"}
    model = load_model(artifact)
    batch = predict_records(load_manifest(holdout), model, AppConfig(quality=_quality()), 1.0, verbose=False)
    assert {item.status for item in batch} == {"uncertain"}


def test_locator_crops_only_unique_box_and_preserves_provided_box(tmp_path):
    image = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(3).integers(50, 175, (100, 120, 3), dtype=np.uint8)).save(image)
    focus = (20.0, 20.0, 90.0, 80.0)
    batch = [Record(str(i), image, str(i), bbox=focus if i == 3 else None) for i in range(4)]

    class Locator:
        def locate_batch(self, images):
            assert len(images) == 3
            return [[(focus, 0.85)], [], [(focus, 0.9), ((1, 1, 12, 10), 0.8)]]

    class Classifier:
        mode = "supervised"

        def predict_images(self, images):
            assert len(images) == 2
            assert all(img.size != (120, 100) for img in images)
            return [Prediction(0.9, 0.1, {}, "open_hatch", "open_hatch")] * 2

    observations = predict_records(batch, Classifier(), AppConfig(quality=_quality()), 0.5,
                                   verbose=False, locator=Locator())
    assert [o.localization["source"] for o in observations] == ["detected", "not_detected", "ambiguous", "provided"]
    assert [o.status for o in observations] == ["finding", "rephoto", "rephoto", "finding"]
    assert observations[0].record.bbox == focus
    assert batch[0].bbox is None


def test_full_frame_fallback_requires_single_target_and_keeps_its_threshold(tmp_path):
    image = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(7).integers(50, 175, (100, 120, 3), dtype=np.uint8)).save(image)
    row = Record("i", image, "asset")

    class EmptyLocator:
        def locate_batch(self, images):
            return [[] for _ in images]

    class Probe:
        mode = "supervised"
        classes = ["normal", "open_hatch"]
        normal_label = "normal"
        threshold = 0.47
        meta = {"crop_mode": "full"}

        def predict_images(self, images):
            return [Prediction(0.53, 0.47, {}, "open_hatch", "open_hatch") for _ in images]

    cfg = AppConfig(quality=_quality())
    with pytest.raises(ValueError, match="одного целевого"):
        predict_records([row], Probe(), cfg, 0.60, locator=EmptyLocator(), full_frame_model=Probe())
    result = predict_records([row], Probe(), cfg, 0.60, locator=EmptyLocator(), single_target=True,
                             full_frame_model=Probe(), verbose=False)[0]
    assert result.status == "finding"
    assert result.localization == {"source": "full_frame", "box_count": 0}
    assert result.record.bbox is None
    assert result.decision_threshold == 0.47


def test_vertex_schema_hides_ground_truth_and_counts_uncertainty_as_miss():
    body = request_body(Image.new("RGB", (60, 60)))
    assert set(body["contents"][0]["parts"][1]["inlineData"]) == {"mimeType", "data"}
    assert "label" in body["generationConfig"]["responseSchema"]["required"]
    mocked = {"candidates": [{"content": {"parts": [
        {"text": "analysis", "thought": True},
        {"text": json.dumps({"label": "uncertain", "visible_evidence": "Dark opening"})},
    ]}}]}
    decoded = decode_response(mocked)
    result = summarize([{**decoded, "label": "open_hatch"}, {"label": "normal", "prediction": "normal"}])
    assert result["coverage"] == 0.5
    assert result["per_class"]["open_hatch"]["recall"] == 0


@pytest.mark.parametrize(
    "reply, source, confirmation",
    [
        ({"status": 200, "prediction": "open_hatch", "bbox": (20, 20, 90, 80)}, "vertex_confirmed", "confirmed_bbox"),
        ({"status": 200, "prediction": "normal", "bbox": (20, 20, 90, 80)}, "full_frame", "local_disagreement"),
        ({"status": 429, "prediction": "uncertain", "error": "RESOURCE_EXHAUSTED"}, "full_frame", "unavailable"),
    ],
)
def test_vertex_recovers_only_locally_confirmed_boxes_and_respects_budget(tmp_path, reply, source, confirmation):
    image = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(8).integers(55, 160, (100, 120, 3), dtype=np.uint8)).save(image)
    rows = [Record(str(i), image, str(i)) for i in range(2)]

    class EmptyLocator:
        def locate_batch(self, images):
            return [[] for _ in images]

    class Probe:
        mode = "supervised"
        classes = ["normal", "open_hatch"]
        normal_label = "normal"
        threshold = 0.47

        def __init__(self, crop_mode):
            self.meta = {"crop_mode": crop_mode}

        def predict_images(self, images):
            assert all((img.size == (120, 100)) == (self.meta["crop_mode"] == "full") for img in images)
            return [Prediction(0.9, 0.1, {}, "open_hatch", "open_hatch") for _ in images]

    class VertexStub:
        model = "gemini-4-argon"

        def inspect(self, img, *, full_frame):
            assert full_frame and img.size == (120, 100)
            return reply

    observations = predict_records(
        rows, Probe("auto"), AppConfig(quality=_quality()), 0.6, locator=EmptyLocator(),
        single_target=True, full_frame_model=Probe("full"), cloud_reviewer=VertexStub(), cloud_limit=1, verbose=False,
    )
    first, second = observations
    assert first.localization["source"] == source
    assert first.cloud_review["confirmation"] == confirmation
    assert [o.status for o in observations] == ["finding", "finding"]
    assert second.cloud_review is None and second.localization["source"] == "full_frame"
    assert (first.record.bbox is not None) == (confirmation == "confirmed_bbox")
