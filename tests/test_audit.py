from pathlib import Path
from types import SimpleNamespace
import csv
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from inspector.classifier import Prediction, _cv_predict, validate_artifact_task
from inspector.config import AppConfig, ModelConfig, PUBLIC_REVIEW_LABEL, PUBLIC_SOURCE
from inspector.evaluate import evaluate_holdout
from inspector.manifest import load_manifest


def write_manifest(directory, rows):
    with (directory / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_keeps_ids_and_converts_only_public_legacy_label(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "p.jpg")
    row = dict(image="p.jpg", image_id="0001", asset_id="NA", criticality="0",
               label="open_hatch", source=PUBLIC_SOURCE)
    write_manifest(tmp_path, [row, dict(row, image_id="0002", source="field")])
    public, field = load_manifest(tmp_path)
    assert (public.image_id, public.asset_id, public.criticality) == ("0001", "NA", 0)
    assert public.label == PUBLIC_REVIEW_LABEL
    assert field.label == "open_hatch"
    write_manifest(tmp_path, [row, row])
    with pytest.raises(ValueError, match="image_id"):
        load_manifest(tmp_path)


def test_grouped_cv_requires_independent_defect_examples():
    with pytest.raises(ValueError, match="независимым объектам"):
        _cv_predict(np.arange(8).reshape(4, 2), np.array(["normal"] * 3 + ["defect"]),
                    np.arange(4), ModelConfig(cv_folds=2), "normal")


def test_old_public_artifact_cannot_emit_defect_advice(tmp_path):
    pd.DataFrame({"source": [PUBLIC_SOURCE]}).to_csv(tmp_path / "oof.csv", index=False)
    with pytest.raises(ValueError, match="void"):
        validate_artifact_task(tmp_path, {"classes": ["normal", "open_hatch"]})
    assert "Проверить" in AppConfig().resolve_sign(PUBLIC_REVIEW_LABEL)["action"]


def test_holdout_crop_metrics_and_all_refusals(tmp_path, monkeypatch):
    artifact = tmp_path / "model"
    artifact.mkdir()
    settings = AppConfig().quality.to_dict()
    settings.update(min_width=1, min_height=1, min_sharpness=0, min_contrast=0)
    (artifact / "meta.json").write_text(json.dumps(dict(quality_config=settings, threshold=.5, crop_mode="bbox")))
    sizes = []
    def predict(images):
        sizes.extend(im.size for im in images)
        return [Prediction(p, 1-p, {}, "x", "x") for p in [.8, .2, .2][:len(images)]]
    monkeypatch.setattr("inspector.evaluate.load_model", lambda _: SimpleNamespace(mode="supervised", predict_images=predict))
    Image.new("RGB", (100, 100), (128, 128, 128)).save(tmp_path / "photo.jpg")
    rows = [dict(image="photo.jpg", asset_id=g, image_id=str(i), label=label, bbox_x1=25, bbox_x2=75,
                 bbox_y1=25, bbox_y2=75) for i,(g,label) in enumerate([("01","x"),("01","x"),("02","normal")])]
    write_manifest(tmp_path, rows)
    r = evaluate_holdout(artifact, tmp_path)
    assert sizes == [(58, 58)] * 3
    assert r["recall"] == .5 and r["recall_assets"] == 1
    assert r["end_to_end_recall_images"] == .5
    Image.new("RGB", (100, 100)).save(tmp_path / "photo.jpg")
    r = evaluate_holdout(artifact, tmp_path)
    assert (r["n_usable"], r["n_refused"], r["coverage"], r["end_to_end_recall_images"]) == (0, 3, 0, 0)
    assert r["recall_ci95"] is None
    pd.DataFrame({"asset_id":["01"]}).to_csv(artifact / "oof.csv", index=False)
    with pytest.raises(ValueError, match="пересекается"):
        evaluate_holdout(artifact, tmp_path)


def test_web_media_decisions_and_original_pixels(tmp_path):
    from fastapi.testclient import TestClient
    from inspector.web import create_app
    from inspector.pipeline import analyze
    from inspector.synth import generate
    from inspector.classifier import train_model
    data, artifact, run = (tmp_path / name for name in ("photos", "model", "run"))
    generate(data, n_assets=12, seed=11)
    train_model(data, artifact, backbone="classical", verbose=False)
    target = load_manifest(data)[0]
    # A one-frame manifest with no bounding box matches the upload contract.
    write_manifest(data, [dict(image=str(target.image_path), asset_id="01", image_id="снимок")])
    (run / "images").mkdir(parents=True)
    sentinel = run / "images" / "notes.txt"
    sentinel.write_text("keep")
    analyze(data, artifact, run, threshold=.25, verbose=False)
    assert sentinel.read_text() == "keep"
    client = TestClient(create_app(run, artifact))
    card = client.get("/api/cards").json()[0]
    assert client.post("/api/cards/missing/decision", json={"decision":"confirmed"}).status_code == 404
    assert client.get("/media/reviews.sqlite3").status_code == 404
    assert client.get("/media/cards.json").status_code == 404
    assert client.get("/media/images/notes.txt").status_code == 404
    uploaded = client.post("/api/analyze", files={"file":("test.jpg",target.image_path.read_bytes(),"image/jpeg")})
    assert uploaded.status_code == 200
    live = uploaded.json()["card"]
    for key in ("status", "p_finding", "threshold", "quality"):
        assert live[key] == card[key]
    assert client.post(f"/api/cards/{card['card_id']}/decision", json={"decision":"confirmed"}).is_success
    assert card["card_id"] in client.get("/api/export.csv").text
    analyze(data, artifact, run, threshold=.6, verbose=False)
    assert client.get("/api/cards").json()[0]["review_status"] == "new"
    assert client.get("/api/summary").json()["review_counts"] == {}


def test_usable_frame_resolves_rephoto_within_same_visit(tmp_path):
    from inspector.classifier import Prediction
    from inspector.manifest import Record
    from inspector.pipeline import Observation, build_cards
    from inspector.quality import Quality

    cfg = AppConfig()
    observations = []
    for image_id, ok in [("кадр-А", True), ("кадр-Б", False)]:
        q = Quality(ok, float(ok), 40, 128, 20, None, 640, 480, [] if ok else ["нерезко"])
        obs = Observation(Record(image_id, tmp_path / "photo", "01"), q,
                          Prediction(.1, .9, {}, "normal", "open_hatch"),
                          "normal" if ok else "rephoto", "normal" if ok else "unusable")
        observations.append(obs)
    card, = build_cards(observations, cfg, .5, include_normal=True)
    assert (card["status"], card["image_id"], card["frames_count"]) == ("normal", "кадр-А", 2)
    assert not build_cards(observations, cfg, .5, include_normal=False)


def test_fetch_refuses_unrelated_output_before_writing_any_split(tmp_path, monkeypatch):
    from scripts import fetch_public_manholes as fetch

    def extract(_zips, output):
        for label in fetch.LABEL_MAP:
            folder = output / "train" / label
            folder.mkdir(parents=True)
            for i in range(3):
                (folder / f"{i}.jpg").write_bytes(f"{label}-{i}".encode())
    monkeypatch.setattr(fetch, "_download", lambda *args: None)
    monkeypatch.setattr(fetch, "_extract", extract)
    test = tmp_path / "hold"
    test.mkdir()
    sentinel = test / "existing"
    sentinel.write_text("keep")
    args = ["--cache", str(tmp_path/"cache"), "--out", str(tmp_path/"train"),
            "--holdout-out", str(test), "--official-out", str(tmp_path/"official")]
    with pytest.raises(ValueError, match="занят"):
        fetch.main(args)
    assert sentinel.read_text() == "keep"
    assert not (tmp_path/"train").exists()


def test_only_normal_model_calibrates_by_asset_and_keeps_custom_class(tmp_path):
    from inspector.classifier import load_model, train_model
    from inspector.config import QualityConfig
    from inspector.evaluate import evaluate_artifact

    rows = []
    rng = np.random.default_rng(3)
    for i in range(4):
        image = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        image.save(tmp_path / f"{i}.png")
        rows.append(dict(image=f"{i}.png", image_id=str(i), asset_id="A" if i < 3 else "B", label="ok"))
    write_manifest(tmp_path, rows)
    artifact = tmp_path / "artifact"
    train_model(tmp_path, artifact, ModelConfig(normal_label="ok"), backbone="classical",
                quality=QualityConfig(min_width=1, min_height=1), verbose=False)
    model = load_model(artifact)
    assert model.mode == "anomaly" and model.normal_label == "ok"
    assert set(model.predict_images([image])[0].probs) == {"ok", "anomaly"}
    assert len(pd.read_csv(artifact / "anomaly_oof.csv")) == 4
    assert "recall" not in evaluate_artifact(artifact)
